from flaskext.mysql import MySQL
import json


class Connection:
    # def __init__(self):
    #     self.MYSQL_DATABASE_HOST = 'database-song.cc03pstew9fp.ap-southeast-1.rds.amazonaws.com'
    #     self.MYSQL_DATABASE_PORT = '3306'
    #     self.MYSQL_DATABASE_USER = 'admin'
    #     self.MYSQL_DATABASE_PASSWORD = 'research19'
    #     self.MYSQL_DATABASE_DB  = 'songs_schema'

    def initiate_connection(self, app):
        app.config["MYSQL_DATABASE_HOST"] = 'database-song.cc03pstew9fp.ap-southeast-1.rds.amazonaws.com'
        app.config["MYSQL_DATABASE_PORT"] =  3306
        app.config["MYSQL_DATABASE_USER"] = 'admin'
        app.config["MYSQL_DATABASE_PASSWORD"] = 'research19'
        app.config["MYSQL_DATABASE_DB"]   = 'songs_schema'

        mysql = MySQL()
        mysql.init_app(app)
        self.cursor = None
        with app.app_context():
            self.cursor = mysql.get_db().cursor()

    def query(self,a):
        query_string = "SELECT * FROM songs_schema.song_table where song_id IN "+ str(tuple(a))+"limit 10;"
        self.cursor.execute(query_string)
        data = self.cursor.fetchall()
        results = self.convert_to_json(data)
        return results

    def convert_to_json(self,data):
        array_of_objects = []
        object = {}
        for i in data:
            d = {}
            d['song_id'] = i[0]
            d['song_name'] = i[1]
            d['song_url'] = i[2]
            d['thumbnail_url'] = i[3]
            d['artist_id'] = i[4]
            array_of_objects.append(d)

        object['suggestions'] = array_of_objects
        return object
        


