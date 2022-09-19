import tensorflow as tf

def fun_train_simclr(model, datain, fun_augment_01, fun_augment_02, 
                     epochs = 100, batch_size = 32, verbose = 1, 
                     shuffle = True, patience = 3):
  
  num_data  = datain.shape[0]
  num_batch = int(num_data//batch_size) # Reminder: // is divide and floor
  z_size    = model.layers[-1].weights[-1].shape[0]
  loss      = []  

  for i0 in range(epochs):
    counter    = 0
    loss_batch = []

    if shuffle:
      ind_shuffle = tf.experimental.numpy.random.randint(0,datain.shape[0],datain.shape[0])
      datain = datain[ind_shuffle,:]


    for i1 in range(num_batch):
      if i1 == num_batch - 1:
        ind_case = range(counter, num_data)
      else:
        ind_case = range(counter, counter + batch_size)


      x_tilda_01  = fun_augment_01(datain[ind_case,:])
      x_tilda_02  = fun_augment_02(datain[ind_case,:]) 

      x_tilda     = tf.reshape(tf.concat([x_tilda_01,x_tilda_02], axis = 1), 
                               (x_tilda_01.shape[0] + x_tilda_02.shape[0],  
                                x_tilda_01.shape[1], x_tilda_01.shape[2], 
                                x_tilda_01.shape[3]))
      
      z_real      = tf.random.uniform((x_tilda.shape[0],z_size)) # dummy variable

      # Train on batch
      var = model.train_on_batch(x_tilda, z_real)
      loss_batch.append(var[0]) 


      counter  = counter + batch_size 

      if verbose:
        if i1 == num_batch - 1:
          print("\r SimCLR | Epoch {:04d}/{:04d} - Batch {:04d}/{:04d} - Loss {:8.5F}".format(i0+1, epochs, i1+1, num_batch, sum(loss_batch)/len(loss_batch)), flush=True)
        else:
          print("\r SimCLR | Epoch {:04d}/{:04d} - Batch {:04d}/{:04d} - Loss {:8.5F}".format(i0+1, epochs, i1+1, num_batch, sum(loss_batch)/len(loss_batch)), end="", flush=True)

      

    loss.append(sum(loss_batch)/len(loss_batch))

    if i0>patience:
      loss_hist_min = min(loss)

      if loss[-1] > loss_hist_min:
        break
  
  return model, loss