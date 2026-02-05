from langchain_core.messages import HumanMessage, AIMessage


a = HumanMessage(content="Hello, How are you?")
b = AIMessage(content="Hi there!, I'm doing alright, you?")
# d = HumanMessage(content="Okay, i want to know how an llm agent works")
e = AIMessage(content="An llm agent works by transformer arch. where it has multiple attention heads...")
# f = HumanMessage(content="Wow that's great tell me more, i'd like to build one someday")
g = AIMessage(content="Okay, so initially when the input is passed in to the model,\
it goes in through what is called positional encoding, this is done to maintain order at which the text came in as input...")

c = []
c.append(a)
c.append(b)
# c.append(c)
# c.append(d)
c.append(e)
# c.append(f)
c.append(g)

print(c, len(c))


# print(hasattr(a, "content"))
print(isinstance(c[-1], AIMessage))
print(HumanMessage)

print("is AIMessage in c: ", AIMessage in c)
for n in range(len(c), 0, -1):
#     print(type(messagetype))
    print(n)


for i in reversed(range(len(c))):
    # print(i)
    print("The output of the list when being iterated backwards is thus: ", c[i], "and it is in the", i, "position.")
    # if isinstance(c[-(i + 1)], HumanMessage):
    #     print("the value of i is", -i)
    #     print(c[-(i + 1)].content)
    #     break
    if isinstance(c[i], HumanMessage):
        print("the value of i is", i)
        print(c[i].content)
        break

for i in range(len(c)):
    print(i)


question = [HumanMessage(content='what is th  recent advancemtn in artifucial intelligence?', additional_kwargs={}, response_metadata={}, id='8bc58868-3e82-4e9c-9fc7-f06dd643fe0d'), AIMessage(content='What is the recent advancement in artificial intelligence?', additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.5-flash', 'safety_ratings': [], 'model_provider': 'google_genai'}, id='lc_run--019becc6-5593-7e50-9566-c0a531cfb4b2-0', usage_metadata={'input_tokens': 519, 'output_tokens': 320, 'total_tokens': 839, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 311}})]
print(question[-1])
print("last rephrased question is ", question[-1].content)


# print("is the content of the last question in ['what', 'recent']", question[-1].content in ["what", "recent"])
for i in ["Hello", "hell", "helen", "hella", "hello"]:
    print(i)