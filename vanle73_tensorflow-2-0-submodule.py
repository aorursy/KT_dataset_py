def train():

    



    #log

    summary_writer = tf.summary.create_file_writer('./tensorboard') 

    #train    

    raw_ds=tf.data.TFRecordDataset('../input/bert-joint-baseline/nq-train.tfrecords')

    #token_map_ds = raw_ds.map(_decode_tokens)

    decoded_ds = raw_ds.map(_decode_record_train)

    #ds = decoded_ds.batch(batch_size=2,drop_remainder=False)

    train_dataset = decoded_ds.shuffle(buffer_size=1000).batch(1)

    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-6)

    manager = tf.train.CheckpointManager(cpkt, directory='./save_2', max_to_keep=2)

    count_step=0

    bar=tqdm.tqdm(cycle(train_dataset))

    for a in bar:

        count_step+=1

        input_mask=a['input_mask']

        input_ids=a['input_ids']

        segment_ids=a['segment_ids']

        start_positions=a['start_positions']

        end_positions=a['end_positions']

        inpanswer_typesut_ids=a['answer_types']

        unique_id=tf.constant([0]*input_ids.shape[0])

        

        with tf.GradientTape() as tape:

            _,start_logits,end_logits,ans_type=model([unique_id,input_ids,input_mask,segment_ids])

            start_loss= compute_loss(start_logits,start_positions,bert_utils.FLAGS.max_seq_length)

            end_loss=compute_loss(end_logits,end_positions,bert_utils.FLAGS.max_seq_length)

            answer_type_loss=compute_label_loss(ans_type,inpanswer_typesut_ids,5)

            total_loss = (start_loss + end_loss + answer_type_loss) / 3.0



            bar.set_description("loss {}".format(total_loss))



            with summary_writer.as_default():                               

                tf.summary.scalar("loss", total_loss, step=count_step)

        optimizer.apply_gradients(grads_and_vars=zip(tape.gradient(total_loss, model.trainable_variables), model.trainable_variables))  
