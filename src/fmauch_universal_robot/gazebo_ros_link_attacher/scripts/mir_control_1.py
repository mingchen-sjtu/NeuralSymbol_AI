#!/usr/bin/env python3
# Created by Charles Kristensen - Mobile Industrial Robots
# Created using Python 3.8.2
# Testen in MiR software version 2.8.3

import requests
import json
from pprint import pprint # Used to print nicely and easy readable

header = {
'Content-Type': 'application/json',
'Authorization': 'Basic YWRtaW46OGM2OTc2ZTViNTQxMDQxNWJkZTkwOGJkNGRlZTE1ZGZiMTY3YTljODczZmM0YmI4YTgxZjZmMmFiNDQ4YTkxOA==' # admin user
}


# Get function to make the REST call based on a URL
def get(url):
    # Sending the REST command
    try:
        response = requests.get(url, headers=header)
    #print("Response code: {}".format(response )) # uncomment to see the response code from the REST api
    except Exception as e:
        print(e)

    # Converting the response from json format to reable text
    status = json.loads(response.content.decode('utf-8'))
    return status


### Start of the program ###
if __name__ == "__main__":
    print("Remember to change the ip!")

# Set the IP of the robot, make sure it is online and it is reachable
    ip = "192.168.12.20" # Ip of the robot

# Creating the URL for the REST api call using the IP from above
    url_mission_scheduler = "http://" + ip + "/api/v2.0.0/statistics/distance"

# Sending the Get request to the fleet
    req = get(url_mission_scheduler)
#pprint(req)

# Adding all the data to the output file
with open("output.txt","a") as myFile:
    for i in range(len(req)):
        myFile.write(str(req[i]['date']) + ",")
        myFile.write(str(req[i]['distance']))
        myFile.write("\n")

    myFile.close()