def enforce_max_instance(max_count=3):
    print(f"\n--- Decorator Configuration: Setting max instance to {max_count}---")
    def class_decorator(cls):
        print(f"--- Decorator: Applying max instance enforcement to {cls.__name__}")
        cls.__max_instance_count = max_count
        cls.__current_instance_count = 0
        original_new = cls.__new__
        original_init = cls.__init__

        def new_new(sub_cls, *args, **kwargs):
            # print(f"current_instance_count: {sub_cls.__current_instance_count}; max_instance_count: {sub_cls.__max_instance_count}")
            # if sub_cls.__current_instance_count >= sub_cls.__max_instance_count:
            #     raise RuntimeError(f"Maximum uninitialized instance created: allowable instance is only {sub_cls.__max_instance_count}")
            
            # return super().__new__(sub_cls)
            instance = original_new(sub_cls)

            # cls.__current_instance_count += 1

            return instance

        def new_init(self, *args, **kwargs):
            # self.current_instance = self.__class__.__current_instance_count
            # self.max_count = self.__class__.__max_instance_count

            # print(f"current_instance_count: {self.current_instance}; max_instance_count: {self.max_count}")

            # if self.current_instance >= self.max_count:
            if self.__class__.__current_instance_count >= self.__class__.__max_instance_count:
                raise RuntimeError(f"Maximum uninitialized instance created: allowable instance is only {self.__class__.__max_instance_count}")
            
            
            original_init(self, *args, **kwargs)

            self.__class__.__current_instance_count += 1
            # print(f"current_instance is now: {self.current_instance}")
        # cls.__current_instance_count += 1
        
        cls.__new__ = new_new
        cls.__init__ = new_init
        
        return cls

    return class_decorator



@enforce_max_instance(max_count=2)
class Car:
    def __init__(self, make, model):
        self.make = make
        self.model = model
        print(f"car {make} {model} initialized...")

    def drive(self):
        print(f"Driving the {self.make} {self.model} on the road.")

# Car = enforce_max_instance()(Car)
# Car = car(Car)

car1 = Car("Toyota", "Camry")
car2 = Car("Lexus", "RX350")


# print(car1, car2)

try:
    car3 = Car("Ford", "Focus")
except Exception as e:
    print(f"Error creating car3: {e}")

# print(f"\n Total car created after attempt: {Car.__current_instance_count}")
print(f"\nTotal cars created so far: {Car.__current_instance_count}")
print(f"\nFree car creation slots remaining before exhaustion: {Car.__max_instance_count - Car.__current_instance_count}")