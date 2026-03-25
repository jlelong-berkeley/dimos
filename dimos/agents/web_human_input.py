# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from threading import Thread
from typing import TYPE_CHECKING, Any

import reactivex as rx
import reactivex.operators as ops

from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.transport import pLCMTransport
from dimos.stream.audio.node_normalizer import AudioNormalizer
from dimos.utils.logging_config import setup_logger
from dimos.web.robot_web_interface import RobotWebInterface

if TYPE_CHECKING:
    from dimos.stream.audio.base import AudioEvent

logger = setup_logger()


class WebInput(Module):
    _web_interface: RobotWebInterface | None = None
    _thread: Thread | None = None
    _human_transport: pLCMTransport[str] | None = None
    _agent_transport: pLCMTransport[Any] | None = None
    _agent_unsub: Any = None
    _agent_response_subject: rx.subject.Subject[str] | None = None

    @rpc
    def start(self) -> None:
        super().start()

        self._human_transport = pLCMTransport("/human_input")
        self._agent_transport = pLCMTransport("/agent")
        self._agent_response_subject = rx.subject.Subject()

        audio_subject: rx.subject.Subject[AudioEvent] = rx.subject.Subject()

        self._web_interface = RobotWebInterface(
            port=5555,
            text_streams={"agent_responses": self._agent_response_subject},
            audio_subject=audio_subject,
        )

        normalizer = AudioNormalizer()

        # Here to prevent unwanted imports in the file.
        from dimos.stream.audio.stt.node_whisper import WhisperNode

        stt_node = WhisperNode()

        # Connect audio pipeline: browser audio → normalizer → whisper
        normalizer.consume_audio(audio_subject.pipe(ops.share()))
        stt_node.consume_audio(normalizer.emit_audio())

        # Subscribe to both text input sources
        # 1. Direct text from web interface
        unsub = self._web_interface.query_stream.subscribe(self._publish_human_text)
        self._disposables.add(unsub)

        # 2. Transcribed text from STT
        unsub = stt_node.emit_text().subscribe(self._publish_human_text)
        self._disposables.add(unsub)

        # 3. Agent output from /agent topic for web text stream panel
        self._agent_unsub = self._agent_transport.subscribe(self._on_agent_message)

        self._thread = Thread(target=self._web_interface.run, daemon=True)
        self._thread.start()

        logger.info("Web interface started at http://localhost:5555")

    @rpc
    def stop(self) -> None:
        if self._agent_unsub:
            self._agent_unsub()
            self._agent_unsub = None
        if self._agent_transport:
            self._agent_transport.stop()
        if self._agent_response_subject:
            self._agent_response_subject.on_completed()
            self._agent_response_subject = None
        if self._web_interface:
            self._web_interface.shutdown()
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._human_transport:
            self._human_transport.stop()
        super().stop()

    def _on_agent_message(self, msg: Any) -> None:
        if self._agent_response_subject is None:
            return
        try:
            text = self._format_agent_message(msg)
            if text:
                self._agent_response_subject.on_next(text)
        except Exception:
            logger.exception("Failed to forward agent message to web text stream")

    @staticmethod
    def _format_agent_message(msg: Any) -> str:
        msg_type = str(getattr(msg, "type", msg.__class__.__name__)).lower()
        content = getattr(msg, "content", "")
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            additional = getattr(msg, "additional_kwargs", None)
            if isinstance(additional, dict):
                tool_calls = additional.get("tool_calls", [])

        lines: list[str] = []
        if content:
            lines.append(str(content))

        if tool_calls:
            lines.append("tool_calls:")
            for tc in tool_calls:
                if isinstance(tc, dict):
                    name = tc.get("name", "unknown")
                    args = tc.get("args", {})
                    lines.append(f"- {name}({args})")
                else:
                    lines.append(f"- {tc}")

        if not lines:
            lines.append("<no response>")

        return f"[{msg_type}] " + "\n".join(lines)

    def _publish_human_text(self, text: Any) -> None:
        if self._human_transport is None:
            return
        cleaned = str(text).strip()
        if cleaned:
            self._human_transport.publish(cleaned)


web_input = WebInput.blueprint

__all__ = ["WebInput", "web_input"]
