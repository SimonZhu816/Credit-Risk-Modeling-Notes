# S1：数据准备
feas = ['']
feas_nan = [f+'_nan' for f in feas]
feas_qcut = [f+'_qcut' for f in feas]
feas_str = 'uid,biz_date'
for f in feas:
  feas_str += f',cast({f} as double) as {f}'
data =spark.sql("selct {feas_str} from feas_table").toPandas()
conf = {}
for f in tqdm(feas):
  data[f] = pd.to_numeric(data[f],errors='coerce',downcast = 'float')
  bins,labels = pd.qcut(data[f],q=16,retbins=True,duplicates='drop')
  data[f + '_qcut'] = pd.to_numeric(pd.cut(data[f],bins=labels,labels=list(range(len(labels)-1)),include_lowest=True),errors='coerce',downcast = 'float')
  conf[f] = [data[f].mean(),data[f].std(),list(labels),data[f+'_qcut'].mean(),data[f+'_qcut'].std()]

with open('conf.txt','wb') as f:
  pickle.dump(conf,file=f)

with open('conf.txt','rb') as f:
  conf = pickle.load(f)

f_str = ',concat('
for f in feas:
  f_str += f'if({f} if null or {round(float(conf[f][1]),4)} = 0,0, ({f} - {round(float(conf[f][0]),4)})/{round(float(conf[f][1]),4)}), "#DD#", '

f_str = f_str[:-10] + ') as f_str'

f_nan_str = ',concat('
for f in feas:
  f_nan_str += f'if({f} is null,1.0,0.0), "#DD#", '
f_nan_str = f_nan_str[:-10] + ') as f_nan_str'

f_nan_qcut = ',concat('
for f in feas:
  qcut = 'case'
  for i in range(len(conf[f][2]) - 1):
    if i == len(conf[f][2]) - 2:
      qcut += f' when {f} >= {round(float(conf[f][2][i]),4) and {f}<= {round(float(conf[f][2][i+1]),4)} then {i}'
    else:
      qcut += f' when {f} >= {round(float(conf[f][2][i]),4) and {f}< {round(float(conf[f][2][i+1]),4)} then {i}'
  qcut += f' else {round(float(conf[f][3]),4)} end'
  f_qcut_str += f'if({f} is null or {round(float(conf[f][4]),4) = 0 or {f} < {round(float(conf[f][2][0]),4)} or {f}>{round(float(conf[2][-1]),4)},0,({qcut}-{round(float(conf[f][3]),4)}) / {round(float(conf[f][4]),4)}),"#DD#", '
f_qcut_str = f_qcut_str[:-10] + ') as f_qcut_str'

feas_str = 'uid,biz_date,label,label2,label_wgt,label2_wgt'
feas_str += f_str
feas_str += f_nan_str
feas_str += f_nan_qcut

# 特征宽表2
create table feas_table_2 as 
select {feas_str}
  from feas_table

feas_str = 'uid,biz_date,train_test,label,label2,label_wgt,label2_wgt'
feas_str += ', split(f_str,"#DD#") as f'
feas_str += ', split(f_nan_str,"#DD#") as f_nan'
feas_str += ', split(f_qcut_str,"#DD#") as f_qcut'
# 特征宽表3
create table feas_table_3 as 
select {feas_str}
  from feas_table_2

# 存储tfrecords文件
select uid,biz_date,train_test,label,label2,label_wgt,label2_wgt,f,f_nan,f_qcut
  from feas_table_3
where train_test = 'train'
r.repartition(10).write.format("tfrecords").mode("overwrite").save("viewfs://hadoop/user/simon/train.tfrecords")
!hadoop dfs -get viewfs://hadoop/user/simon/train.tfrecords dnn/train.tfrecords

select uid,biz_date,train_test,label,label2,label_wgt,label2_wgt,f,f_nan,f_qcut
  from feas_table_3
where train_test = 'testA'
r.repartition(10).write.format("tfrecords").mode("overwrite").save("viewfs://hadoop/user/simon/testA.tfrecords")
!hadoop dfs -get viewfs://hadoop/user/simon/testA.tfrecords dnn/testA.tfrecords

select uid,biz_date,train_test,label,label2,label_wgt,label2_wgt,f,f_nan,f_qcut
  from feas_table_3
r.repartition(10).write.format("tfrecords").mode("overwrite").save("viewfs://hadoop/user/simon/dataset.tfrecords")
!hadoop dfs -get viewfs://hadoop/user/simon/dataset.tfrecords dnn/dataset.tfrecords
                                    
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

batch_size = 1024
tfrecords_train = tfrecords_reader_dataset("dnn/train.tfrecords",batch_size=batch_size)
tfrecords_val = tfrecords_reader_dataset("dnn/testA.tfrecords",batch_size=batch_size)

gc.collect()
class AutoInt(Layer):
  def __init__(self,dim=16,n_head=2,**kwargs):
    self.ndims = dim
    self.n_head = n_head
    super(AutoInt,self).__int__(**kwargs)
  def build(self,input_shape):
    assert len(input_shape) ==3, "Input embedding feature map must be 3_D tensor."
    input_dim = input_shape[-1]
    self.W_Q = []
    self.W_K = []
    self.W_V = []
    # 构建多个attention
    for i in range(self.n_head):
      # 初始化W_Q、W_K、W_V
      self.W_Q.append(K.variable(tf.random.trancated_normal(shape=(input_dim,self.ndims)), name = f"AI_query_{self.name}_"+str(i)))
      self.W_K.append(K.variable(tf.random.trancated_normal(shape=(input_dim,self.ndims)), name = f"AI_key_{self.name}_"+str(i)))
      self.W_V.append(K.variable(tf.random.trancated_normal(shape=(input_dim,self.ndims)), name = f"AI_value_{self.name}_"+str(i)))
    self.w_res = K.variable(tf.random.trancated_normal(shape=(input_dim,self.ndims*self.n_head)), name = f"AI_w_res_{self.name}_"+str(i))
    super(AutoInt,self).build(input_shape) # 一定要在最后调用它
  def call(self,x):
    attention_heads = []
    for i in range(self.n_head):
      # 映射到d维空间
      embed_q = tf.mutul(x,self.W_Q[i])
      embed_k = tf.mutul(x,self.W_K[i])
      embed_v = tf.mutul(x,self.W_V[i])
      # 计算attention
      energy = tf.mutul(embed_q,tf.transpose(embed_k,[0,2,1]))
      attention = tf.nn.softmax(energy)
      attention_output = tf.mutul(attention,embed_v)
      attention_heads.append(attention_output)
    # concat multi head
    multi_attention_output = K.concatenate(attention_heads,axis=-1)
    # ResNet
    output = multi_attention_output + tf.matmul(x,self.w_res)
    return K.relu(output)
  def compute_output_shape(self,input_shape):
    output_shape = list(input_shape)
    output_shape[-1] = self.ndims * self.n_head
    return (output_shape)
  def get_config(self):
    config = {'ndims':self.ndims, 'n_head':self.n_head}
    base_config = super(AutoInt,self).get_config()
    base_config.update(config)
    return base_config

class NoneLayer(Layer):
  def __init__(self,**kwargs):
    super(NoneLayer,self).__int__(**kwargs)
  def build(self,input_shape):
    # 为该层创建一个可训练的权重
    self.kernel = self.add_weight(name = 'kernel',
                                  shape = (1,input_shape[1]),
                                  initializer = 'uniform',
                                  trainable = True)
    super(NoneLayer,self).__int__(**kwargs)
  def call(self,x):
    return tf.math.multiply(x,self.kernel)
  def compute_output_shape(self,input_shape):
    return (input_shape[0])
class WeightLayer(Layer):
  def __init__(self,**kwargs):
    super(WeightLayer,self).__int__(**kwargs)
  def build(self,input_shape):
    # 为该层创建一个可训练的权重
    self.kernel = self.add_weight(name = 'kernel',
                                  shape = (input_shape[-2],16),
                                  initializer = 'uniform',
                                  trainable = True)
    super(WeightLayer,self).__int__(**kwargs)
  def call(self,x):
    return tf.math.multiply(x,self.kernel)
  def compute_output_shape(self,input_shape):
    return (input_shape[0])
    
ipt = tf.keras.layers.Input(shape=(len(feas),))
ipt_nan = tf.keras.layers.Input(shape=(len(feas),))
ipt_bin = tf.keras.layers.Input(shape=(len(feas),))

x1 = NoneLayer()(ipt)
x2 = NoneLayer()(ipt_nan)
x3 = NoneLayer()(ipt_bin)
x = layers.concatenate([layers.add([x1,x2]),layers.add([x2,x3])])
x = layers.BatchNormalization(x)
wide = x

deep = layers.Dense(64,activation="relu")(x)
deep = layers.BatchNormalization(deep)
deep = layers.Dense(32,activation="relu")(deep)
deep = layers.BatchNormalization(deep)
deep = layers.Dense(16,activation="relu")(deep)

x_autoint = WeightLayer()(keras.layers.Reshape([x.shape[-1],1])(x))
x_autoint = AutoInt(dim=16,n_head=2,)(x_autoint)
x_autoint = keras.layers.Reshape([x_autoint.shape[-1] * x_autoint.shape[-2]])(x_autoint)
x_autoint = keras.layers.Dense(x.shape[-1],activation='relu',name=f'autoint_dense')(keras.layers.Dropout(0.2)(x_autoint))

all_x = layers.concatenate([wide,x_autoint,deep])

x = layers.Dropout(0.5)(all_x)
x = layers.Dense(512,activation='relu')(x)
x = layers.BatchNormalization(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256,activation='relu')(x)
x = layers.BatchNormalization(x)
x = layers.Dropout(0.5)(x)
label = layers.Dense(1,activation='sigmoid',name="label")(x)

x = layers.Dropout(0.5)(all_x)
x = layers.Dense(512,activation='relu')(x)
x = layers.BatchNormalization(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256,activation='relu')(x)
x = layers.BatchNormalization(x)
x = layers.Dropout(0.5)(x)
label2 = layers.Dense(1,activation='sigmoid',name="label2")(x)

model = tf.keras.Model(inputs=[ipt,ipt_nan,ipt_bin],outputs=[label,label2])
model.compile(keras.optimizers.Adam(1e-3),
              loss = ['binary_crossentropy' for _ in range(2)],
              loss_weights = [1 for i in range(2],
              metrics =[tf.keras.metrics.AUC(name='auc') for _ in range(2))
def scheduler(epoch,lr):
  if epoch < 3:
    return 1e-3
  if epoch % 3 == 0:
    return min(1e-3,1.2*lr)
  if epoch % 10 == 0:
    return 1e-3/2
  return max(lr*0.8,1e-4)
callbacks = [
  ModelCheckpoint("checkpoint.h5",monitor='val_loss',mode='min',save_best_only=True),
  EarlyStopping(monitor="label_auc",patience=5,mode='max'),
  LearningRateScheduler(scheduler,verbose=1),
  ReduceLROnPlateau(monitor="val_loss",factor=0.8,patience=3,verbose=1,mode='min',min_lr=1e-6)
]
model.fit(
  tfrecords_train,
  epochs=30,
  callbacks=callbacks,
  validation_data = tfrecords_val,
  max_queue_size=0,
  steps_per_epoch = 405856 // batch_size,
  validation_steps = 101043 // batch_size,
  workers = 0,
  use_multiprocessing = False,
  verbose =2 ,
  shuffle =False
)
model.save("baseline.h5")
initializer = tf.keras.initializers.RandomNormal(mean=0,stddev=0.05,seed=2024)

model.load_weights("baseline.h5")
model.save("baseline.tf",save_format = 'tf')
plot_model(model,show_shapes=True)

model_save_path = os.getcwd() + "baseline.tf"
!ls {model_save_path}
!save_model_cli show --dir {model_save_path} --tag_set serve --signature_def serving_default


# S3：评估单元
batch_size = 1024
tfrecords_test = tfrecords_reader_dataset("dnn/dataset.tfrecords",repeat=False,batch_size=batch_size,is_test=True)
uid = []
biz_date = []
prob = []
prob2 = []
for s in tqdm(tfrecords_test.__iter__()):
  pred = model.predict(s[0])
  prob += list(pred[0].reshape(-1))
  prob2 += list(pred[1].reshape(-1))
  uid += [str(a,encoding='utf-8') for a in s[1].numpy().reshape(-1)]
  biz_date += [str(a,encoding='utf-8') for a in s[2].numpy().reshape(-1)]
pred  = pd.DataFrame(
    {
      'uid':uid,
      'biz_date':biz_date,
      'prob':prob,
      'prob2':prob2,
    }
  )

pred.index = pred[['uid','biz_date']]
pred = pred[['prob','prob2']]
pred.to_pickle("res_dnn.pickle")
