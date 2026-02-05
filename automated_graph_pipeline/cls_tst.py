class MyDecorator:
    # def __new__(cls, arg1, arg2):
    #     cls_name = cls.__name__
    #     cls_instance = super().__new__(cls)
    #     return cls_instance, cls_name
    
    @classmethod
    def class_name(cls):
        # return cls.__name__
        print(f"The name of this class is {cls.__name__}")
    
    def __init__(self, arg1, arg2=None):
        self.arg1 = arg1
        self.arg2 = arg2

    def __call__(self, func):
        print("this is an instance:", self)
        def wrapper( *args, **kwargs):
            print(f"\n -- Before calling {self.__class__.__name__} with decorator arguments: arg1 = {self.arg1}, arg2 = {self.arg2}")
            print(f"Function: {func.__name__} called with arguments: {args}, {kwargs}")
            result = func(*args, **kwargs)
            print(f"After calling {func.__name__}, result {result}")
            return result
        
        return wrapper


# @MyDecorator("Epic")
def greet(name):
    return f"Hello, {name}"

instance = MyDecorator(arg1="Epic", arg2="shit")
obj = instance(greet)
obj("Gloria")
