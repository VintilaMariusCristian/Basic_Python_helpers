#! /usr/bin/python

# Imports
import requests

def send_simple_message():
    print("I am sending an email.")
    return requests.post(
        "https://api.mailgun.net/v3/sandboxc5b43439f39040a7b6528f627b99f9d1.mailgun.org/messages",
        auth=("api", "fc0a956fc427e9a978309173ec9ea6eb-1553bd45-6669c5d9"),
        data={"from": 'Licenta Andreea Elena Circeag <andreeaelena1898@gmail.com>',
            "to": ["andreeaelenac18@gmail.com"],
            "subject": "You have a visitor",
            "html":  " is at your door.  </html>"})
                      
request = send_simple_message()
print ('Status: '+format(request.status_code))
print ('Body:'+ format(request.text))
