# S1：数据准备

# S2：训练单元
feas = ['']
feas_nan = [f+'_nan' for f in feas]
feas_qcut = [f+'_qcut' for f in feas]

def parse_example(serialized_example):
  expected_features= {
    "uid":tf.io.FixedLenFeature([1],dtype=tf.string),
    "biz_date":tf.io.FixedLenFeature([1],dtype=tf.string),
    "label":tf.io.FixedLenFeature([1],dtype=tf.int64),
    "label2":tf.io.FixedLenFeature([1],dtype=tf.int64),
    "label_wgt":tf.io.FixedLenFeature([1],dtype=tf.int64),
    "label2_wgt":tf.io.FixedLenFeature([1],dtype=tf.int64),
    "f":tf.io.FixedLenFeature(len(feas),dtype=tf.int64),
    "f_nan":tf.io.FixedLenFeature(len(feas),dtype=tf.int64),
    "f_qcut":tf.io.FixedLenFeature(len(feas),dtype=tf.int64)
  }
  example = tf.io.parse_single_example(serialized_example,expected_features)
  return (example["f"],example["f_nan"],example["f_qcut"]),\
         (example["label"],example["label2"]),\
         (example["label_wgt"][0],example["label2_wgt"][0])
def parse_example_test(serialized_example):
  expected_features = {
    "uid":tf.io.FixedLenFeature([1],dtype=tf.string),
    "biz_date":tf.io.FixedLenFeature([1],dtype=tf.string),
    "f":tf.io.FixedLenFeature(len(feas),dtype=tf.int64),
    "f_nan":tf.io.FixedLenFeature(len(feas),dtype=tf.int64),
    "f_qcut":tf.io.FixedLenFeature(len(feas),dtype=tf.int64)
  example = tf.io.parse_single_example(serialized_example,expected_features)
  return (example["f"],example["f_nan"],example["f_qcut"]),example["uid"],example["biz_date"]
  }
def tfrecords_reader_dataset(filename,repeat=True,n_readers=2,batch_size=1024,n_parse_threads=2,shuffle_buffer_size=1024*10,is_test=False):
  filenames = [(filename + "/" + name) for name in os.listdir(filename) if name.startswith("part")]
  dataset = tf.data.Dataset.list_files(filenames)
  if repeat:
    dataset = dataset.repeat()
  dataset = dataset.interleave(
    lambda filename:tf.data.TFRecordDataset(filename,compression_type=None),
    cycle_length = n_readers
  )
  dataset.shuffle(shuffle_buffer_size)
  if is_test:
    dataset = dataset.map(parse_example_test,num_parallel_calls = n_parse_threads)
  else:
    dataset = dataset.map(parse_example,num_parallel_calls = n_parse_threads)
  dataset = dataset.batch(batch_size)
  return dataset

# S3：评估单元
