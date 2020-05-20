import redis

class RedisDB:
    
    def __init__(self):
        self.redis_db = redis.Redis(host='redis-db', port=6379, decode_responses=True)

    def insert(self, key, value):
        self.redis_db.rpush(key, value)

    def get(self, key, index):
        return self.redis_db.lindex(key, index)

    def list_len(self, key):
        return self.redis_db.llen(key)
