# import pymysql

# db = pymysql.connect("1.1.1.1", "root", "password", port=1)


def read_sql(db, sql_command={}, verbose=False):
    cursor = db.cursor()
    cursor.execute("SHOW databases;")
    results = cursor.fetchall()
    cursor.execute(f"USE {results[1][0]};")
    cursor.execute("SHOW tables;")
    results = cursor.fetchall()
    if verbose:
        print(f"tables {results}")
    if isinstance(sql_command, str):
        sql = sql_command
    else:
        columns = ",".join(sql_command.get("columns", ["*"]))
        table = sql_command.get("table", "")
        conditions = sql_command.get("conditions", "")
        if conditions:
            sql = "SELECT {} FROM {} WHERE {}".format(
                columns, table, conditions)
        else:
            sql = "SELECT {} FROM {}  ".format(columns, table)

    if verbose:
        print(f"sql commands {sql}")

    cursor.execute(sql)
    for col in cursor.description:
        print(col)

    results = cursor.fetchall()
    db.close()
    headers = [c[0] for c in cursor.description]
    return results, headers
