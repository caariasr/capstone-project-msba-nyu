
#Functions we need
from pyspark.sql.functions import *
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import re


def findIterDel(string):
  pattern = 'MLD(?:(MM|MCU|MCD))+MLU(?:((BSDBSU|REDREU)+|(SDSU|EDEU)|[^A-Z]D))'
  pattern_comp = re.compile(pattern)
  list_index = []
  iter = pattern_comp.finditer(string)
  for i in iter:
    list_index.append(str(i.start()))
    list_index.append(str(i.end()))
    return '|'.join(list_index)


def findIterMov(string):
  pattern = '(?:(MM|MCU|MCD))+'
  pattern_comp = re.compile(pattern)
  list_index = []
  iter = pattern_comp.finditer(string)
  for i in iter:
    list_index.append(str(i.start()))
    list_index.append(str(i.end()))
    return '|'.join(list_index)


def findIterDelKey(string):
  pattern = '(BSDBSU|REDREU)+'
  pattern_comp = re.compile(pattern)
  list_index = []
  iter = pattern_comp.finditer(string)
  for i in iter:
    list_index.append(str(i.start()))
    list_index.append(str(i.end()))
    return '|'.join(list_index)


def findIterMovKey(string):
  pattern = '(ANDANU|LDLU|ASDASU|AEDAEU)+'
  pattern_comp = re.compile(pattern)
  list_index = []
  iter = pattern_comp.finditer(string)
  for i in iter:
    list_index.append(str(i.start()))
    list_index.append(str(i.end()))
    return '|'.join(list_index)


def isInIter(value, text):
  array = text.split('|')
  if len(array) == 0:
    return 0
  check = 0
  for i in range(1, len(array), 2):
    if value > int(array[(i - 1)]) and value < int(array[i]):
      check = 1
    return check

def regex_countPy(pattern, string):
  return len(re.findall(pattern, string))


# Register them as udf

addzeroes = spark.udf.register("stringLengthString",
                               lambda x: "0" * (6 - len(x)))
finditerdel = spark.udf.register("findIterWithPythonDel",
                                 lambda x: findIterDel(x))
finditermov = spark.udf.register("findIterWithPythonDel",
                                 lambda x: findIterMov(x))
finditerdelkey = spark.udf.register("findIterWithPythonDelKey",
                                    lambda x: findIterDelKey(x))
finditermovkey = spark.udf.register("findIterWithPythonDelKey",
                                    lambda x: findIterMovKey(x))
checkindex = spark.udf.register("checkIndexWithPython",
                                lambda x, y: isInIter(x, y))
strlen = spark.udf.register("stringLengthString", lambda x: len(x))
regexp_count = spark.udf.register("regexp_count", regex_countPy)


# Read remote postgresql tables

remote_table = spark.read.format("jdbc")\
  .option("driver", dbutils.widgets.get("driver"))\
  .option("url", dbutils.widgets.get("url"))\
  .option("dbtable", dbutils.widgets.get("table"))\
  .option("user", dbutils.widgets.get("user"))\
  .option("password", dbutils.widgets.get("password"))\
  .option("lowerBound", dbutils.widgets.get("lowerBound"))\
  .option("upperBound", dbutils.widgets.get("upperBound"))\
  .option("numPartitions", dbutils.widgets.get("numPartitions"))\
  .option("partitionColumn", dbutils.widgets.get("partitionColumn"))\
  .option("fetchSize", dbutils.widgets.get("fetchSize"))\
  .load()

remote_table_codes = spark.read.format("jdbc")\
  .option("driver", dbutils.widgets.get("driver"))\
  .option("url", dbutils.widgets.get("url"))\
  .option("dbtable", dbutils.widgets.get("tablecodes"))\
  .option("user", dbutils.widgets.get("user"))\
  .option("password", dbutils.widgets.get("password"))\
  .load()


# Load tables into parquet format to the cluster

remote_table.write.format("parquet").saveAsTable("keys")
remote_table_codes.write.format("parquet").saveAsTable("keycodes")


# load new registered table as pyspark dataframe

keys = spark.sql("SELECT * FROM keys")


# START FEATURE ENGINEERING

keys = keys.withColumn(
  "editor",
  when(keys["app"].contains("- Eclipse") |
       keys["app"].contains("- Spring Tool Suite"), "eclipse")\
  .otherwise(
    when(keys["app"].contains("- Visual Studio Code"), "vscode")\
    .otherwise(
      when(keys["app"].contains("- NetBeans"), "netbeans")\
      .otherwise(
        when(keys["app"].contains("- Sublime Text"), "sublime text")\
        .otherwise(
          when(keys["app"].contains("- Notepad++"), "notepad++")\
          .otherwise("other")
        )
      )
    )
  ))

keys = keys.withColumn(
  "keyval",
  when(keys["keyval"].isNull(), "mouse")\
  .otherwise(
    concat_ws(" - ", col("device"), col("keyval"))))

key_codes = spark.sql("SELECT * FROM keycodes")
keys = keys.filter((keys["time"] > '2017-11-27') & (keys["editor"] != 'other'))
keys_recoded = keys.join(key_codes, ["keyval"], 'left')


keys_recoded = keys_recoded.select(
  col("user_id_id").alias("user_id"), col("email"), col("company"),
  col("date_joined"), col("time"), col("device"),
  year("time").alias("year"),
  month("time").alias("month"),
  dayofmonth("time").alias("day"),
  hour("time").alias("hours"),
  minute("time").alias("minutes"),
  second("time").alias("seconds"), col("x_coord"), col("y_coord"),
  col("keyboard_language"), col("editor"), col('event_type'),
  col("new").alias("event")).filter(col('event') != 'DELETE')


# Calculate time difference between keystrokes of same editor
my_window = Window.partitionBy().orderBy("id")
keys_recoded = keys_recoded.orderBy("user_id", "editor", "time")
keys_recoded = keys_recoded.withColumn("id", F.monotonically_increasing_id())
keys_recoded = keys_recoded.withColumn("milliseconds",
                                       regexp_extract(
                                           keys_recoded.time.cast("string"),
                                           "(\.)(\d+)", 2))
keys_recoded = keys_recoded.withColumn(
  "zeroes",
  addzeroes(keys_recoded.milliseconds.cast("string")).cast("string"))
keys_recoded = keys_recoded.withColumn(
  "milliseconds",
  concat_ws('', keys_recoded.milliseconds.cast("string"),
            keys_recoded.zeroes).cast("float"))
keys_recoded = keys_recoded.withColumn("milliseconds",
                                       col("milliseconds") * 0.000001)
keys_recoded = keys_recoded.withColumn(
  "time_diff",
  (keys_recoded.time.cast("bigint") + keys_recoded.milliseconds -
   lag(keys_recoded.time.cast("bigint") + keys_recoded.milliseconds, 1).over(
     Window.partitionBy("user_id", "editor")
     .orderBy("user_id", "editor", "time"))).cast("float"))
keys_recoded = keys_recoded.na.fill(0)


# Define a task run as consecutive keystrokes with less than 40 secs with no key
# being pressed

keys_recoded = keys_recoded.withColumn(
  "time_diff",
  when(keys_recoded["time_diff"].isNull(), 0)\
  .otherwise(keys_recoded["time_diff"]))
keys_recoded = keys_recoded.withColumn(
  "is_new_task_run", (keys_recoded['time_diff'] > 40).cast("integer"))
windowval = (Window.partitionBy("user_id", "editor").orderBy('time')
             .rangeBetween(Window.unboundedPreceding, 0))
keys_recoded = keys_recoded.withColumn(
  'task_runs',
  F.sum(keys_recoded['is_new_task_run']).over(windowval) + 1)

# define the keystroke and event code. For example dU is the letter d being
# released

keys_recoded = keys_recoded.withColumn(
  "key_event",
  when(col("event_type") == "up", concat(col("event"), lit('U')))\
  .otherwise(when(col("event_type") == "down",
                  concat(col("event"), lit('D'))).otherwise(col("event"))
             ))
keys_recoded = keys_recoded.withColumn(
  'last_index',
  F.sum(strlen(col('key_event'))).over(
    Window.partitionBy("user_id", "editor", "task_runs")
    .orderBy("user_id", "editor", "task_runs", "time")).cast("float") - 1)


# First checkpoint (Write the new table in parquet format)

keys_recoded.write.format("parquet").saveAsTable("keys_recoded")


# Load Checkpoint (Uncomment)
# keys_recoded = spark.sql("SELECT * FROM keys_recoded")

# SECOND STAGE: AGREGATE KEYSTROKE EVENTS AS STRINGS. For example,
# 'LCDvDvULCU' is pressing left ctrl then v then releasing v and finally
# releasing ctrl

string_data = keys_recoded.groupby("user_id", "editor", "task_runs").agg(
  concat_ws('', F.collect_list(keys_recoded.key_event)).alias('key_string'),
  count(keys_recoded.key_event).alias('string_length'),
  F.first(keys_recoded.time).alias('first_time'),
  F.last(keys_recoded.time).alias('last_time')).orderBy('task_runs')

string_data = string_data.withColumn('task_del_pos',
                                     finditerdel(col('key_string')))
string_data = string_data.withColumn('task_del_pos_key',
                                     finditerdelkey(col('key_string')))
string_data = string_data.withColumn('task_mov_pos',
                                     finditermov(col('key_string')))
string_data = string_data.withColumn('task_mov_pos_key',
                                     finditermovkey(col('key_string')))

string_data.write.format("parquet").saveAsTable("string_data")

# SECOND CHECKPOINT STRING DATA (UNCOMMENT)
# string_data = spark.sql("SELECT * FROM string_data")

string_data = string_data.withColumn(
  'Command Code',
  when(
    col('editor') == 'name of the editor',
    regexp_count(
      lit('regular expression for the command'),
      col('key_string'))).otherwise(0))

string_data_for_join = string_data.drop("first_time", "last_time")

keys_joined = keys_recoded.join(
    string_data_for_join, ['user_id', 'editor', 'task_runs'], 'left').drop(
        'is_new_task_run', 'time_diff', 'zeroes', 'event', 'event_type')

keys_joined = keys_joined.withColumn(
  'is_in_task_del_event',\
  when((checkindex(col('last_index'), col('task_del_pos_key')) == 1) |
       (checkindex(col('last_index'), col('task_del_pos')) == 1), 1)\
  .otherwise(0))
keys_joined = keys_joined.withColumn(
  'is_in_task_del_mouse_event',
  checkindex(
    col('last_index'),
    col('task_del_pos')))
keys_joined = keys_joined.withColumn(
  'is_in_task_mov_event',\
  when(((checkindex(col('last_index'), col('task_mov_pos_key')) == 1) |
        (checkindex(col('last_index'), col('task_mov_pos')) == 1)), 1)\
  .otherwise(0))
keys_joined = keys_joined.withColumn(
  'is_in_task_mov_mouse_event',\
  checkindex(col('last_index'), col('task_mov_pos')))

keys_joined = keys_joined.withColumn(
    "is_group_del", (keys_joined['diff_del'] != 0).cast("integer"))
keys_joined = keys_joined.withColumn(
    "is_group_del_mouse", (keys_joined['diff_del_mouse'] != 0).cast("integer"))
keys_joined = keys_joined.withColumn(
    "is_group_mov", (keys_joined['diff_mov'] != 0).cast("integer"))
keys_joined = keys_joined.withColumn(
    "is_group_mov_mouse", (keys_joined['diff_mov_mouse'] != 0).cast("integer"))
windowval = (Window.partitionBy("user_id", "editor",
                                "task_runs").orderBy('time').rangeBetween(
                                    Window.unboundedPreceding, 0))
keys_joined = keys_joined.withColumn(
    'group_del',
    F.sum(keys_joined['is_group_del']).over(windowval) + 1)
keys_joined = keys_joined.withColumn(
    'group_del_mouse',
    F.sum(keys_joined['is_group_del_mouse']).over(windowval) + 1)
keys_joined = keys_joined.withColumn(
    'group_mov',
    F.sum(keys_joined['is_group_mov']).over(windowval) + 1)
keys_joined = keys_joined.withColumn(
    'group_mov_mouse',
    F.sum(keys_joined['is_group_mov_mouse']).over(windowval) + 1)


# DELETS HAS ALL DELETES, DELETES WITH MOUSE ONLY THE ONES WITH MOUSE

del_keys = keys_joined.filter((col('is_in_task_del_event') == 1) | (
  col('is_in_task_del_mouse_event') == 1)).select(
    'user_id', 'editor', 'id', 'key_event', 'time', 'task_runs',
    'milliseconds')
del_keys_mouse = keys_joined.filter(
  col('is_in_task_del_mouse_event') == 1).select('user_id', 'editor', 'id',
                                                 'key_event', 'time',
                                                 'task_runs', 'milliseconds')
mov_keys = keys_joined.filter((col('is_in_task_mov_event') == 1) | (
  col('is_in_task_mov_mouse_event') == 1)).select(
    'user_id', 'editor', 'id', 'key_event', 'time', 'task_runs',
    'milliseconds')
mov_keys_mouse = keys_joined.filter(
  col('is_in_task_mov_mouse_event') == 1).select('user_id', 'editor', 'id',
                                                 'key_event', 'time',
                                                 'task_runs', 'milliseconds')
del_keys = del_keys.withColumn('dummy', lit(1))
windowval = (Window.partitionBy("user_id", "editor", "task_runs").orderBy(
    "user_id", "editor", "task_runs", "time").rangeBetween(
        Window.unboundedPreceding, 0))
dummyval = (Window.partitionBy("dummy").orderBy("user_id", "editor",
                                                "task_runs", "time")
            .rangeBetween(Window.unboundedPreceding, 0))
del_keys = del_keys.withColumn(
    "consecutive2", (col('id') - lag(col('id'), 1).over(
        Window.partitionBy("user_id", "editor", "task_runs")
        .orderBy("user_id", "editor", "task_runs", "time"))))
del_keys = del_keys.na.fill(0)
del_keys = del_keys.withColumn("is_consecutive",
                               (col('consecutive2') != 1).cast("integer"))
del_keys = del_keys.withColumn('consecutive_groups',
                               F.sum(col('is_consecutive')).over(dummyval))

del_keys_mouse = del_keys_mouse.withColumn('dummy', lit(1))
del_keys_mouse = del_keys_mouse.withColumn(
    "consecutive2", (col('id') - lag(col('id'), 1).over(
        Window.partitionBy("user_id", "editor", "task_runs")
        .orderBy("user_id", "editor", "task_runs", "time"))))
del_keys_mouse = del_keys_mouse.na.fill(0)
del_keys_mouse = del_keys_mouse.withColumn(
    "is_consecutive", (col('consecutive2') != 1).cast("integer"))
del_keys_mouse = del_keys_mouse.na.fill(0)
del_keys_mouse = del_keys_mouse.withColumn(
    'consecutive_groups',
    F.sum(col('is_consecutive')).over(dummyval))

mov_keys = mov_keys.withColumn('dummy', lit(1))

mov_keys = mov_keys.withColumn(
    "consecutive2", (col('id') - lag(col('id'), 1).over(
        Window.partitionBy("user_id", "editor", "task_runs")
        .orderBy("user_id", "editor", "task_runs", "time"))))
mov_keys = mov_keys.na.fill(0)
mov_keys = mov_keys.withColumn("is_consecutive",
                               (col('consecutive2') != 1).cast("integer"))
mov_keys = mov_keys.withColumn('consecutive_groups',
                               F.sum(col('is_consecutive')).over(dummyval))
mov_keys_mouse = mov_keys_mouse.withColumn('dummy', lit(1))
mov_keys_mouse = mov_keys_mouse.withColumn(
    "consecutive2", (col('id') - lag(col('id'), 1).over(
        Window.partitionBy("user_id", "editor", "task_runs")
        .orderBy("user_id", "editor", "task_runs", "time"))))
mov_keys_mouse = mov_keys_mouse.na.fill(0)
mov_keys_mouse = mov_keys_mouse.withColumn(
    "is_consecutive", (col('consecutive2') != 1).cast("integer"))
mov_keys_mouse = mov_keys_mouse.withColumn(
    'consecutive_groups',
    F.sum(col('is_consecutive')).over(dummyval))
del_cons = del_keys.groupBy(
    "user_id", "editor", "task_runs", "consecutive_groups").agg(
        (F.last(col('time').cast("bigint") + col('milliseconds')) -
         F.first(col('time').cast("bigint") + col('milliseconds'))
         ).alias('total_time'))
del_cons = del_cons.withColumn("penalty_del",
                               F.when((col('total_time') - 1.1) >= 0,
                                      (col('total_time') - 1.1)).otherwise(0))

del_cons_mouse = del_keys_mouse.groupBy(
    "user_id", "editor", "task_runs", "consecutive_groups").agg(
        (F.last(col('time').cast("bigint") + col('milliseconds')) -
         F.first(col('time').cast("bigint") + col('milliseconds'))
         ).alias('total_time'))
del_cons_mouse = del_cons_mouse.withColumn(
    "penalty_del_mouse",
    F.when((col('total_time') - 1.1) >= 0,
           (col('total_time') - 1.1)).otherwise(0))

mov_cons = mov_keys.groupBy(
    "user_id", "editor", "task_runs", "consecutive_groups").agg(
        (F.last(col('time').cast("bigint") + col('milliseconds')) -
         F.first(col('time').cast("bigint") + col('milliseconds'))
         ).alias('total_time'))
mov_cons = mov_cons.withColumn("penalty_mov",
                               F.when((col('total_time') - 1.1) >= 0,
                                      (col('total_time') - 1.1)).otherwise(0))

mov_cons_mouse = mov_keys_mouse.groupBy(
    "user_id", "editor", "task_runs", "consecutive_groups").agg(
        (F.last(col('time').cast("bigint") + col('milliseconds')) -
         F.first(col('time').cast("bigint") + col('milliseconds'))
         ).alias('total_time'))
mov_cons_mouse = mov_cons_mouse.withColumn(
    "penalty_mov_mouse",
    F.when((col('total_time') - 1.1) >= 0,
           (col('total_time') - 1.1)).otherwise(0))

del_groups = del_cons.groupBy("user_id", "editor", "task_runs").agg(
    F.sum(col('penalty_del')).alias('penalty_del'))
del_groups_mouse = del_cons_mouse.groupBy(
    "user_id", "editor", "task_runs").agg(
        F.sum(col('penalty_del_mouse')).alias('penalty_del_mouse'))
mov_groups = mov_cons.groupBy("user_id", "editor", "task_runs").agg(
    F.sum(col('penalty_mov')).alias('penalty_mov'))
mov_groups_mouse = mov_cons_mouse.groupBy(
    "user_id", "editor", "task_runs").agg(
        F.sum(col('penalty_mov_mouse')).alias('penalty_mov_mouse'))

full_data = string_data.join(del_groups, ['user_id', 'editor', 'task_runs'],
                             'left')
full_data = full_data.withColumn("milliseconds_first",
                                 regexp_extract(
                                     full_data.first_time.cast("string"),
                                     "(\.)(\d+)", 2))
full_data = full_data.withColumn(
    "zeroes_first",
    addzeroes(full_data.milliseconds_first.cast("string")).cast("string"))
full_data = full_data.withColumn(
    "milliseconds_first",
    concat_ws('', full_data.milliseconds_first.cast("string"),
              full_data.zeroes_first).cast("float"))
full_data = full_data.withColumn("milliseconds_first",
                                 col("milliseconds_first") * 0.000001)
full_data = full_data.withColumn("milliseconds_last",
                                 regexp_extract(
                                     full_data.last_time.cast("string"),
                                     "(\.)(\d+)", 2))
full_data = full_data.withColumn(
  "zeroes_last",
  addzeroes(full_data.milliseconds_last.cast("string")).cast("string"))
full_data = full_data.withColumn(
  "milliseconds_last",
  concat_ws('', full_data.milliseconds_last.cast("string"),
            full_data.zeroes_last).cast("float"))
full_data = full_data.withColumn("milliseconds_last",
                                 col("milliseconds_last") * 0.000001)
full_data = full_data.withColumn(
  'task_time',
  col('last_time').cast("bigint") + col('milliseconds_last') -
  col('first_time').cast("bigint") - col('milliseconds_first'))
full_data = full_data.withColumn('percentage_del',
                                 100 * (col('penalty_del') / col('task_time')))
full_data = full_data.drop('milliseconds_last', 'milliseconds_first',
                           'zeroes_first', 'zeroes_last')
full_data = full_data.filter(col('task_time') > 0)

full_data = full_data.join(del_groups_mouse,
                           ['user_id', 'editor', 'task_runs'], 'left')
full_data = full_data.withColumn('percentage_del_mouse', 100 *
                                 (col('penalty_del_mouse') / col('task_time')))

full_data = full_data.join(mov_groups_mouse,
                           ['user_id', 'editor', 'task_runs'], 'left')
full_data = full_data.withColumn('percentage_mov_mouse', 100 *
                                 (col('penalty_mov_mouse') / col('task_time')))
full_data = full_data.filter(col('task_time') > 0)


# Export data ready for modeling

full_data.coalesce(1).write.csv("final_data_cleaned.csv")
