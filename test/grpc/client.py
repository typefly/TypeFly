import grpc
import string_concat_pb2
import string_concat_pb2_grpc

def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = string_concat_pb2_grpc.StringConcatServiceStub(channel)
    
    response = stub.Concat(string_concat_pb2.StringRequest(a="Hello, ", b="gRPC!"))
    print(f"Received: {response.result}")

if __name__ == '__main__':
    run()