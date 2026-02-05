import request
import json

response = request.get("https://api.stackexchange.com/2.3/questions?order=desc&sort=activity&site=stackoverflow")

print(response)