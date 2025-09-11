import os
import pymongo

from pymongo import MongoClient

"""
    DB Util for MongoDB client
    
    Requirement
    ===========
    Expected: DB Name is configured in the environment variable (~/.bashrc or ~/.bash_profile, or ~/.zshrc)
    For environment variable are required:
    Name of the database:  os.environ['MONGO_DB_NAME']
    DB username: os.environ['MONGO_USER_NAME']
    DB user associated password: os.environ['MONGO_PASSWORD'] # default is "127.0.0.1"
    DB configured port number: os.environ['MONGO_IP'] # default is "27017"
    
    Setup example
    =============
    
    Step 1: Configure a MongoDB Install in the local computer 
    Ref: https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/
    
    In command line run the below command after installation. 
    sudo service mongod start && mongod (Starts the service)
    >> mogosh  (this opens the terminal)
    
    >> use honeysat_log (creates a database)
    >> db.example.insert({"data":{'1': 'one'}}) (creates a sample collection)
    
    >> db.createUser(
        {
            user: "honeysat_user",
            pwd: "h0n4yS&t!",
            roles: [ { role: "readWrite", db: "honeysat_log" } ]
        }
      )
    
    # The above command creates user with authentication
    # exit out of terminal and relogin with the credentials created by running the following command in the terminal
    
    >> mongosh --port 27017 -u "honeysat_user" --authenticationDatabase "honeysat_log" -p 
    >> show collections (see list of collections / tables)
    
    # The above lets use login to the credential provided


    Step 2: Configure environment variable with the below settings
    
    Example configuration to add in  ~/.bash_profile. You can use editior such as vi/vim/nano to add the configuration
        
        export MONGO_DB_NAME = "honeysat_log"
        export MONGO_USER_NAME="honeysat_user"
        export MONGO_PASSWORD="h0n4yS&t!"
        export MONGO_IP="127.0.0.1"
        export MONGO_PORT="27017"

    Step 3: After the completion of step 1 and step 2, the below code is configured to store / retrieve the data.

"""


class MongoDBActor:
    def __init__(self, collection_name, db_name='honeysat_log'):
        self.collection_name = collection_name
        self.db_name = db_name
        self.connection_string = "mongodb://{}:{}@{}:{}/?authSource={}".format(
            os.environ['MONGO_USER_NAME'],
            os.environ['MONGO_PASSWORD'],
            os.environ['MONGO_IP'],
            os.environ['MONGO_PORT'],
            os.environ['MONGO_DB_NAME']
        )
        self.client = MongoClient(self.connection_string)
        self.db = self.client[self.db_name]
        self.col_name = self.db[self.collection_name]

    def insert_data(self, data):
        _insert = self.col_name.insert_one(data)
        return _insert.inserted_id

    # note this does not store the key, it will replace with new data
    def replace_insert_if_not_found(self, key, data, _upsert=True):
        replace = self.col_name.replace_one(key, data, upsert=_upsert)
        return replace.upserted_id

    # this will store the key, and will have a key as well in data
    def find_and_modify(self, key, data):
        update = self.col_name.update_one(key, {'$set': data}, upsert=True)
        return update.upserted_id

    # find one based on the search type: latest or previous result of inserted data
    def find_one(self, param, sort_by=None):
        if sort_by:
            return self.col_name.find_one(param, sort=[(sort_by, pymongo.DESCENDING)])
        else:
            return self.col_name.find_one(param)

    # unique data select
    def distinct(self, key, filter=None):
        if filter:
            _found = self.col_name.distinct(key=key, filter=filter)
        else:
            _found = self.col_name.distinct(key=key)
        if None in _found:
            _found.remove(None)
        if "" in _found:
            _found.remove("")
        return _found

    # find all matching candidates based on key and limit
    def find(self, key=None, limit=None):
        if key and limit is None:
            found = self.col_name.find(key)
        elif key is None and limit:
            found = self.col_name.find({}).limit(limit)
        elif key and limit:
            found = self.col_name.find(key).limit(limit)
        elif key is None and limit is None:
            found = self.col_name.find({})
        else:
            raise Exception("query not supported")
        return found
