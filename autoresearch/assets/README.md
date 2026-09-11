# Assets

这里不再保存外部大模型 checkpoint。

当前只保留静态注册信息，用来说明 `autoresearch/` 内置了哪些方法：

- 从头训练的小型 Transformer
- 本地手工特征 reward model
- 随机初始化的进化搜索器

真正训练出来的权重统一写到 `../runs/`。
