class Human:
    def __init__(self, name, gender):
        self.name = name
        self.gender = gender


    def choose_category(self):
        if self.gender == "male":
            print("I am a man, I must cater, and protect my family.")
        elif self.gender == "female":
            print("I am a woman, I must nourish and preserve my household as a whole.")
        else:
            print("Sorry, wrong input your input should either be a `male` or  `female`.")

    def get_biological_makeup(self):
        if self.gender == "male":
            print(f"Hi, i am {self.name},  My kinds are mostly muscular in nature.")
        else:
            print(f"Hi, I am {self.name}, My kinds are most plumpy in nature.")
        

H1 = Human("Tunde", "male")
print(H1.__dir__())