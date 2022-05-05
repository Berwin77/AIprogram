# Item2Vec(Word2Vec)


- 将input_pt中路径替换为rating.csv所在本机地址

- 将output_pt中路径替换为希望输出地址


   * @param input_pt     输入路径
   * @param timeName     时间戳列名
   * @param itemName     item列名
   * @param userName     user列名
   * @param filter       过滤条件  没有则为""   按照spark sql规范  如 "a >= 0.3"
   * @param embLength    Word2vec中的生成embedding隐向量维度 默认值为8
   * @param windowsSise  Word2vec中的滑动窗口的长度 默认值为5
   * @param output_pt    输出路径

