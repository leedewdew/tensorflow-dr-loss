def dr_loss(logits, targets, num_classes):
    '''
        logits:  [1, 80, 80, 105] -> [num_anchors, 7]
        targets: [1, 80, 80, 15]  -> [num_anchors, 1]
    '''
    L = 6.0
    tau = 4.0
    margin = 0.5
    pos_lambda = 1
    neg_lambda = 0.184

    # num_classes = 80
    N = tf.shape(logits)[0]
    logits  = tf.reshape(logits, [-1, num_classes])  # [h*w*a, 7]
    targets = tf.reshape(targets, [-1, 1])  # [h*w*a, 1]

    # class_range = tf.range(1, num_classes + 1, dtype=targets.dtype)
    ## line 552 in anchors.py
    class_range = tf.range(0, num_classes, dtype=targets.dtype)
    class_range = tf.expand_dims(class_range, axis=0)

    pos_ind  = tf.equal(targets, class_range)
    neg_ind  = tf.not_equal(targets, class_range)
    pos_prob = tf.nn.sigmoid(tf.boolean_mask(logits, pos_ind))
    neg_prob = tf.nn.sigmoid(tf.boolean_mask(logits, neg_ind))

    neg_q    = tf.nn.softmax(neg_prob / neg_lambda, axis=0)
    neg_dist = tf.reduce_sum(neg_q * neg_prob)

    def true_fn():
        pos_q    = tf.nn.softmax(-pos_prob / pos_lambda, axis=0)
        pos_dist = tf.reduce_sum(pos_q * pos_prob)
        loss = tau * tf.math.log(1. + tf.math.exp(L * (neg_dist - pos_dist + margin))) / L
        return loss
    def false_fn():
        loss = tau * tf.math.log(1. + tf.math.exp(L * (neg_dist - 1. + margin))) / L
        return loss
    numel = tf.size(pos_prob)
    loss = tf.cond(numel > 0, true_fn, false_fn)
    loss /= tf.cast((numel + N), tf.float32)

    return loss