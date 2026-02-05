import datetime

class Logger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            print(f"-- Creating a new Logger Insance for the First Time. --")
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance.log_file = "app.log"

            with open(cls._instance.log_file, 'a') as f:
                f.write(f"[{datetime.datetime.now()}] Logger initialized.")

        return cls._instance

    def __init__(self, log_message):
        print("We are now in the init function after the instance has been created but yet to be instantiated...")
        self.log_message = log_message




log1 = Logger("welcome, new user this would be the only log you'd be seeing...")

print(log1.log_message)

# log2 = Logger("So this is a continuation of the log message")
# print(log2.log_message)

