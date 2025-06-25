import pymysql
from database import MyDB

# MyDB class 생성
mydb = MyDB()

# table 생성 쿼리문
create_user = """
    create table
    if not exists
    user (
    id varchar(32) primary key,
    password varchar(64) not null,
    name varchar(32)
    )
"""
# 쿼리문 실행
mydb.sql_query(create_user)
# db server에 동기화하고 연결을 종료
mydb.commit_db()