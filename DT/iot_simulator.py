import warnings, time, random, json, argparse, yaml
import paho.mqtt.client as mqtt
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

def on_connect(client, userdata, flags, rc, properties=None):
    print(f"[MQTT SIMULATOR] Connected with result code {rc}")

def load_configuration(config_path):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"[WARNING] Config file '{config_path}' not found. Using defaults.")
        return {}
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}")
        return {}

parser = argparse.ArgumentParser(description='IoT Traffic Simulator for Digital Twin')
parser.add_argument("--mode", choices=["normal", "attack"], default="normal")
parser.add_argument("--devices", type=int, default=10)
parser.add_argument("--interval", type=float, default=2.0)
parser.add_argument("--duration", type=float)
parser.add_argument("--config", type=str, default="/app/config/experiment_config.yaml")
args = parser.parse_args()


config = load_configuration(args.config)
baseline_duration = config.get('entropy', {}).get('baseline_duration', 300)


if args.duration is not None:
    chosen_duration = args.duration
elif args.mode == "normal":
    chosen_duration = baseline_duration
    print(f"[SIMULATOR] Using configured 'baseline_duration' ({baseline_duration}s) for normal traffic.")
else:
    chosen_duration = float('inf')

print(f"[SIMULATOR] Starting in '{args.mode}' mode for {chosen_duration}s (interval: {args.interval}s, devices: {args.devices})")

client = mqtt.Client(client_id="simulator", protocol=mqtt.MQTTv5)
client.on_connect = on_connect
client.connect("mqtt-broker", 1883)
client.loop_start()

start_time = time.time()
packet_count = 0

try:
    while (time.time() - start_time) < chosen_duration:
        for i in range(1, args.devices + 1):
            device_id = f"device_{i}"
            if args.mode == "normal":
                value = random.uniform(40, 70)
                size = random.randint(60, 200)
            else:
                value = random.uniform(80, 100)
                size = random.randint(400, 1200)

            payload = {
                "device": device_id,
                "src": f"192.168.1.{i}",
                "dst": "10.0.0.1",
                "proto": "UDP",
                "port": 123,
                "size": size,
                "value": value,
                "ts": int(time.time() * 1000)
            }
            topic = f"iot/sensor_data/{device_id}"
            client.publish(topic, json.dumps(payload))
            packet_count += 1
        time.sleep(args.interval)

    print(f"[SIMULATOR] Completed {chosen_duration} second(s) of operation in '{args.mode}' mode.")
    print(f"[SIMULATOR] Total packets sent: {packet_count}")

except KeyboardInterrupt:
    print("\n[SIMULATOR] Stopped by user (Keyboard Interrupt).")
finally:
    client.loop_stop()
    client.disconnect()