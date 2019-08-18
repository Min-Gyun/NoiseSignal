import pymysql as db
import pandas as pd
import logging
logging.basicConfig()
logger = logging.getLogger('logger')

conn = db.connect(host='localhost',
                             user='root',
                             password='1120',
                             db='edgar_db',
                             charset='utf8',
                             autocommit=True,
                             cursorclass=db.cursors.DictCursor)

cursor = conn.cursor()

#SQL = 'CREATE TABLE states (id INT NOT NULL PRIMARY KEY AUTO_INCREMENT, state CHAR(25), population INT(9));'
#SQL = "INSERT INTO states (id, state, population) VALUES (NULL,'Alabama',4822023), (NULL, 'Alaska', '731449'), (NULL, 'Arizona', '6553255'), (NULL, 'Arkansas', '2949131');"
#SQL = "SELECT * FROM states"

def create_table_master_idx():
    ## CREATE MASTER_IDX TABLE
    SQL = "CREATE TABLE master_idx (" \
          "year INT, " \
          "qtr CHAR(4), " \
          "CIK INT, " \
          "name CHAR(32), " \
          "form_type CHAR(8), " \
          "date_filed CHAR(10), " \
          "file_name CHAR(64)," \
          "UNIQUE (file_name));"
    return SQL

def insert_table_master_idx(record):
    SQL = "INSERT INTO master_idx VALUES (" + str(record[0]) + ",'" + record[1] + "'," + str(record[2]) + ",'" + record[3] + "','" + record[4] + "','" + record[5] + "','" + record[6] + "');"

    return SQL

record = [2016, 'QTR1', 1326801, 'Facebook Inc', '10-K', '2016-01-28', 'edgar/data/1326801/0001326801-16-000043.txt']

#SQL = create_table_master_idx()

#SQL = insert_table_master_idx(record)

SQL = "SELECT * FROM master_idx"

try:
    # initiate temp tables
    cursor.execute(SQL)
    result = cursor.fetchall()
    print result

except Exception as e:
    logger.error(e)
    logger.exception(e)
    pass

finally:
    # resources closing
    conn.commit()
    cursor.close()
    conn.close()


