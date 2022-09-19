import tensorflow as tf

def fun_simclr_loss(z_real, z_estimate):
  del z_real

  # Temperature parameter, which is a hyper-parameter to be optimized for a 
  # particular problem 
  toe = .1 

  num = int(z_estimate.shape[0]) #num = 2N

  # Create i & j indices against each other
  ind0 = tf.repeat(tf.expand_dims(tf.range(0,num),axis = 0),num, axis = 0)
  ind1 = tf.reshape(ind0, (num**2,1))[:,0]
  ind2 = tf.reshape(tf.transpose(ind0), (num**2,1))[:,0]

  del ind0

  # Arrange the z_estimate values based on ind1
  vector_1   = tf.gather(z_estimate, ind1, axis = 0)
  del ind1

  # Arrange the z_estimate values based on ind2
  vector_2   = tf.gather(z_estimate, ind2, axis = 0)
  del ind2

  # Compute the cosine similarity of vector_1 and vector_2
  s      = - tf.reshape(tf.keras.losses.cosine_similarity(vector_1, vector_2, axis=1),(num,num))

  del vector_1 
  del vector_2

  # Compute the nominator of l(i,j)
  nom    = tf.exp(s/toe)

  # Compute the denominator of l(i,j)
  x1    = tf.exp(s/toe)

  del s

  x2    = 1-tf.eye(num, dtype = tf.float32)

  
  denom = tf.repeat(tf.expand_dims(tf.math.reduce_sum(x1 * x2, axis = 1), axis = 1), num, axis = 1)
  
  del x1
  del x2

  # Compute l(i,j) for all i and j
  l     = -tf.math.log(nom/denom)

  del nom
  del denom 

  # Compute L
  ind_2k0 = tf.range(0,num,2, dtype=tf.int32) 
  ind_2k1 = tf.range(1,num,2, dtype=tf.int32)

  loss_mat1_1 = tf.gather(l,           ind_2k0, axis = 0)
  loss_mat1_2 = tf.gather(loss_mat1_1, ind_2k1, axis = 1)
  loss_mat1   = tf.linalg.diag_part(loss_mat1_2)

  loss_mat2_1 = tf.gather(l,           ind_2k1, axis = 0)
  loss_mat2_2 = tf.gather(loss_mat2_1, ind_2k0, axis = 1)
  loss_mat2   = tf.linalg.diag_part(loss_mat2_2)

  del l

  loss_mat = loss_mat1 + loss_mat2

  L = tf.math.reduce_sum(loss_mat)/num

  return L