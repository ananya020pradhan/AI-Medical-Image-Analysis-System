def login(username, password):
    users = {
        "doctor": "1234",
        "admin": "admin"
    }
    return users.get(username) == password