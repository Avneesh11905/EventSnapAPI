import asyncio
import httpx
import base64
import time
import grpc

from presentation.grpc.proto import inference_pb2, inference_pb2_grpc


def get_real_image():
    """Read a real image from the EventSnapAPI images directory."""
    image_path = r"d:\EventSnap\EventSnapAPI\images\PXL_20241116_133746627.jpg"
    with open(image_path, "rb") as f:
        return f.read()


async def test_http(base_url="http://localhost:5000"):
    print(f"\n--- Testing HTTP API at {base_url} ---")
    image_bytes = get_real_image()
    b64_str = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "inputs": [b64_str],
        "parameters": {"max_faces": "0", "detection_conf": 0.5, "nms_threshold": 0.4},
    }

    async with httpx.AsyncClient() as client:
        start_time = time.perf_counter()
        try:
            response = await client.post(f"{base_url}/", json=payload, timeout=30.0)
            elapsed = time.perf_counter() - start_time

            if response.status_code == 200:
                data = response.json()
                print(f"HTTP Success in {elapsed:.3f}s!")
                faces = data.get("batch_faces", [[]])[0]
                print(f"   Found {len(faces)} faces in the image.")
            else:
                print(f"HTTP Error: {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"HTTP Connection failed: {e}")


async def test_grpc(target="localhost:50051"):
    print(f"\n--- Testing gRPC API at {target} ---")
    image_bytes = get_real_image()

    start_time = time.perf_counter()
    try:
        async with grpc.aio.insecure_channel(target) as channel:
            stub = inference_pb2_grpc.FaceInferenceStub(channel)
            request = inference_pb2.InferenceRequest(
                images=[image_bytes], detection_conf=0.5, nms_threshold=0.4, max_faces=0
            )

            response = await stub.ExtractFaces(request, timeout=30.0)
            elapsed = time.perf_counter() - start_time

            print(f"gRPC Success in {elapsed:.3f}s!")
            faces = response.results[0].faces
            print(f"   Found {len(faces)} faces in the image.")
    except grpc.RpcError as e:
        print(f"gRPC Error: {e.code()}")
        print(e.details())
    except Exception as e:
        print(f"gRPC Connection failed: {e}")


async def main():
    print("Sending dummy image to live servers to verify they are responding...")
    await test_http()
    await test_grpc()


if __name__ == "__main__":
    asyncio.run(main())
