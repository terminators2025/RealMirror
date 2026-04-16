#!/bin/bash
# Generate Python code from proto file

python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. inference.proto

echo "Proto files generated successfully!"
