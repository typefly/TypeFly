import grpc
from concurrent import futures
import string_concat_pb2
import string_concat_pb2_grpc

class StringConcatService(string_concat_pb2_grpc.StringConcatServiceServicer):
    def Concat(self, request, context):
        return string_concat_pb2.StringResponse(
            result=request.a + request.b
        )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    string_concat_pb2_grpc.add_StringConcatServiceServicer_to_server(StringConcatService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()