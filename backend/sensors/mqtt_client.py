"""Optional MQTT bridge: subscribes to a topic carrying the same JSON as POST /api/sensors."""
from __future__ import annotations

import json
import logging

log = logging.getLogger("mqtt")


class MqttBridge:
    def __init__(self, on_data):
        self.on_data = on_data
        self.client = None
        self.status = "disabled"

    def start(self, host: str, port: int, topic: str):
        self.stop()
        try:
            import paho.mqtt.client as mqtt
            c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

            def on_connect(client, _u, _f, rc, _p=None):
                client.subscribe(topic)
                self.status = f"connected to {host}:{port}"

            def on_message(_c, _u, msg):
                try:
                    self.on_data(json.loads(msg.payload.decode()), "mqtt")
                except Exception:  # noqa: BLE001
                    log.warning("bad MQTT payload on %s", msg.topic)

            c.on_connect, c.on_message = on_connect, on_message
            c.connect_async(host, port, 30)
            c.loop_start()
            self.client, self.status = c, "connecting"
        except Exception as e:  # noqa: BLE001
            self.status = f"error: {e}"
            log.warning("MQTT start failed: %s", e)

    def stop(self):
        if self.client is not None:
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception:  # noqa: BLE001
                pass
            self.client = None
        self.status = "disabled"
