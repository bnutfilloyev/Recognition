from locust import HttpUser, task

passport = "AA0000445"


class Recognizer(HttpUser):
    @task
    def recognize(self):
        res = self.client.post("/image/{}".format(passport), files={"image": open("test.jpeg", "rb")})
        print(res.text)
