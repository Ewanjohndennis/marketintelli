def to_jsonrpc(methods, params, request_id=1):
    return{
        "jsonrpc": "2.0",
        "method": methods,
        "params": params,
        "id": request_id
    }
data={"a": 5, "b":7}
rpc=to_jsonrpc("add_numbers", data)
print(rpc)