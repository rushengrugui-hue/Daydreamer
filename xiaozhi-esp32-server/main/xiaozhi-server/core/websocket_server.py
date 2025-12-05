import asyncio
import json
import websockets
import os
import uuid

from config.logger import setup_logging
from core.connection import ConnectionHandler
from config.config_loader import get_config_from_api
from core.auth import AuthManager, AuthenticationError
from core.utils.modules_initialize import initialize_modules
from core.utils.util import check_vad_update, check_asr_update

# === 核心导入：使用 DTO 构建标准消息 ===
from core.providers.tts.dto.dto import TTSMessageDTO, SentenceType, ContentType

TAG = __name__

# ---【配置】定义指令 ---
COMMAND_MAP = {
    "forward": "收到，全速前进！",
    "backward": "注意，正在倒车。",
    "left": "正在左转。",
    "right": "正在右转。",
    "stop": "紧急停止！",
    "mode_control": "进入遥控模式"
}

class WebSocketServer:
    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logging()
        self.config_lock = asyncio.Lock()
        
        # === 1. 初始化模块 (必须开启 TTS) ===
        modules = initialize_modules(
            self.logger, self.config,
            "VAD" in self.config["selected_module"],
            "ASR" in self.config["selected_module"],
            "LLM" in self.config["selected_module"],
            "TTS" in self.config["selected_module"], # 必须 True
            "Memory" in self.config["selected_module"],
            "Intent" in self.config["selected_module"],
        )
        
        self._vad = modules.get("vad")
        self._asr = modules.get("asr")
        self._llm = modules.get("llm")
        self._intent = modules.get("intent")
        self._memory = modules.get("memory")
        self._tts = modules.get("tts") 

        auth_config = self.config["server"].get("auth", {})
        self.auth_enable = auth_config.get("enabled", False)
        self.allowed_devices = set(auth_config.get("allowed_devices", []))
        secret_key = self.config["server"]["auth_key"]
        expire_seconds = auth_config.get("expire_seconds", None)
        self.auth = AuthManager(secret_key=secret_key, expire_seconds=expire_seconds)
        
        self.esp32_ws = None      
        self.esp32_handler = None 

    async def start(self):
        server_config = self.config["server"]
        host = server_config.get("ip", "0.0.0.0")
        port = int(server_config.get("port", 8000))

        async with websockets.serve(
            self._handle_connection, host, port, process_request=self._http_response
        ):
            await asyncio.Future()

    async def _handle_connection(self, websocket):
        headers = dict(websocket.request.headers)
        device_id = headers.get("device-id", None)

        if device_id is None:
            from urllib.parse import parse_qs, urlparse
            request_path = websocket.request.path
            if not request_path: await websocket.close(); return
            parsed_url = urlparse(request_path)
            query_params = parse_qs(parsed_url.query)
            if "device-id" not in query_params:
                await websocket.send("端口正常，如需测试连接，请使用test_page.html")
                await websocket.close()
                return
            else:
                device_id = query_params["device-id"][0]
                websocket.request.headers["device-id"] = device_id

        # ===【手机 App 连接处理】===
        if device_id == "mobile_app": 
            self.logger.bind(tag=TAG).info("手机 App 已连接！")
            try:
                async for message in websocket:
                    self.logger.bind(tag=TAG).info(f"收到 App 原始数据: {message}")
                    
                    try:
                        cmd_data = json.loads(message)
                        target_cmd = None

                        # 解析鸿蒙 App 指令
                        msg_type = cmd_data.get("type")       
                        direction = cmd_data.get("direction") 

                        if msg_type == "move":
                            if direction == "forward": target_cmd = "forward"
                            elif direction == "backward": target_cmd = "backward"
                        elif msg_type == "turn":
                            if direction == "left": target_cmd = "left"
                            elif direction == "right": target_cmd = "right"
                        elif msg_type == "stop": target_cmd = "stop"
                        
                        if target_cmd is None: target_cmd = cmd_data.get("cmd")

                        self.logger.bind(tag=TAG).info(f"翻译后的指令: {target_cmd}")

                        if target_cmd in COMMAND_MAP:
                            reply_text = COMMAND_MAP[target_cmd]
                            
                            # 检查小智是否在线
                            if self.esp32_ws and self.esp32_handler:
                                try:
                                    self.logger.bind(tag=TAG).info(f"指令 [{target_cmd}] 插队，执行【核弹级】清空流程...")

                                    # 1. 服务端内部打断 (停止 LLM 生成)
                                    if hasattr(self.esp32_handler, "_handle_text_message"):
                                        await self.esp32_handler._handle_text_message('{"type": "abort"}')
                                    
                                    # 2. 硬件侧打断 (发送闭嘴指令)
                                    await self.esp32_ws.send(json.dumps({"type": "abort"}))
                                    await self.esp32_ws.send(json.dumps({"type": "stop"}))
                                    
                                    # 3. 【核心步骤】暴力清空 TTS 队列
                                    # 这一步确保“旧账”彻底清除，不会再续接
                                    if self.esp32_handler.tts:
                                        q_text = self.esp32_handler.tts.tts_text_queue
                                        q_audio = self.esp32_handler.tts.tts_audio_queue
                                        
                                        # 循环清空文本队列
                                        while not q_text.empty():
                                            try: q_text.get_nowait()
                                            except: pass
                                        
                                        # 循环清空音频队列
                                        while not q_audio.empty():
                                            try: q_audio.get_nowait()
                                            except: pass
                                            
                                        self.logger.bind(tag=TAG).info("已清空所有旧对话缓存")

                                    # 4. 手动解除封锁 (Abort 会锁住状态，必须解开)
                                    self.esp32_handler.client_abort = False
                                    
                                    # 稍微缓冲，给系统一点喘息时间
                                    await asyncio.sleep(0.1) 

                                    # 5. 推送新指令 (三步走策略，确保稳如老狗)
                                    if hasattr(self.esp32_handler, "tts") and self.esp32_handler.tts:
                                        session_id = uuid.uuid4().hex
                                        self.esp32_handler.sentence_id = session_id 

                                        # (A) FIRST 包：初始化状态，这会告诉 base.py "这是新的一句话，跟前面没关系"
                                        msg_reset = TTSMessageDTO(
                                            sentence_id=session_id,
                                            sentence_type=SentenceType.FIRST, 
                                            content_type=ContentType.TEXT,
                                            content_detail="" 
                                        )
                                        self.esp32_handler.tts.tts_text_queue.put(msg_reset)

                                        # (B) MIDDLE 包：真正的指令文本
                                        msg_text = TTSMessageDTO(
                                            sentence_id=session_id,
                                            sentence_type=SentenceType.MIDDLE,
                                            content_type=ContentType.TEXT,
                                            content_detail=reply_text
                                        )
                                        self.esp32_handler.tts.tts_text_queue.put(msg_text)

                                        # (C) LAST 包：强制立即发送，不等待标点
                                        msg_end = TTSMessageDTO(
                                            sentence_id=session_id,
                                            sentence_type=SentenceType.LAST,
                                            content_type=ContentType.TEXT,
                                            content_detail=""
                                        )
                                        self.esp32_handler.tts.tts_text_queue.put(msg_end)

                                        self.logger.bind(tag=TAG).info(f"已推入新指令: {reply_text}")
                                    else:
                                        self.logger.bind(tag=TAG).error("TTS 模块不可用")

                                    await websocket.send(json.dumps({"status": "sent", "text": reply_text}))
                                except Exception as send_err:
                                    self.logger.bind(tag=TAG).error(f"发送失败: {send_err}")
                            else:
                                self.logger.bind(tag=TAG).warning("小智不在线")
                                await websocket.send(json.dumps({"status": "error", "msg": "小智不在线"}))
                        else:
                            self.logger.bind(tag=TAG).warning(f"无法识别指令: {message}")

                    except Exception as e:
                        self.logger.bind(tag=TAG).error(f"解析出错: {e}")
            except Exception as e:
                self.logger.bind(tag=TAG).info(f"App 断开: {e}")
            return

        # ===【小智连接】===
        self.esp32_ws = websocket 
        self.logger.bind(tag=TAG).info(f"小智已上线: {device_id}")

        try:
            await self._handle_auth(websocket)
        except AuthenticationError:
            await websocket.send("认证失败")
            await websocket.close()
            return
            
        handler = ConnectionHandler(
            self.config, self._vad, self._asr, self._llm, self._memory, self._intent, self
        )
        self.esp32_handler = handler 
        
        try:
            await handler.handle_connection(websocket)
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"连接错误: {e}")
        finally:
            if self.esp32_ws == websocket:
                self.esp32_ws = None
                self.esp32_handler = None
                self.logger.bind(tag=TAG).info("小智已下线")
            try:
                await websocket.close()
            except: pass

    async def _http_response(self, websocket, request_headers):
        if request_headers.headers.get("connection", "").lower() == "upgrade": return None
        return websocket.respond(200, "Server is running\n")

    async def update_config(self) -> bool:
        try:
            async with self.config_lock:
                new_config = get_config_from_api(self.config)
                if new_config is None: return False
                update_vad = check_vad_update(self.config, new_config)
                update_asr = check_asr_update(self.config, new_config)
                self.config = new_config
                modules = initialize_modules(
                    self.logger, new_config, update_vad, update_asr,
                    "LLM" in new_config["selected_module"],
                    "TTS" in new_config["selected_module"], # True
                    "Memory" in new_config["selected_module"],
                    "Intent" in new_config["selected_module"],
                )
                if "vad" in modules: self._vad = modules["vad"]
                if "asr" in modules: self._asr = modules["asr"]
                if "llm" in modules: self._llm = modules["llm"]
                if "intent" in modules: self._intent = modules["intent"]
                if "memory" in modules: self._memory = modules["memory"]
                if "tts" in modules: self._tts = modules["tts"] # True
                
                self.logger.bind(tag=TAG).info(f"更新配置任务执行完毕")
                return True
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"更新服务器配置失败: {str(e)}")
            return False

    async def _handle_auth(self, websocket):
        if self.auth_enable:
            headers = dict(websocket.request.headers)
            device_id = headers.get("device-id", None)
            client_id = headers.get("client-id", None)
            if self.allowed_devices and device_id in self.allowed_devices: return
            else:
                token = headers.get("authorization", "")
                if token.startswith("Bearer "): token = token[7:]
                else: raise AuthenticationError("Missing or invalid Authorization header")
                auth_success = self.auth.verify_token(token, client_id=client_id, username=device_id)
                if not auth_success: raise AuthenticationError("Invalid token")