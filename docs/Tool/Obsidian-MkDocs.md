 obsidian中的图片路径是传统的markdown不一样，使用MkDocs部署的时候有显示问题，用了这个[mkdocs-obsidian-bridge插件](https://github.com/GooRoo/mkdocs-obsidian-bridge)完美解决，讨论可见[这里](https://github.com/GooRoo/mkdocs-obsidian-bridge/issues/17#issuecomment-2369535324)
 
```shell
pip install mkdocs-obsidian-bridge
```

在`mkdocs.yml`里面修改

```YAML
plugins:
  - obsidian-bridge

markdown_extensions:
  - obsidian_media_mkdocs  # this thing embeds audio/video/YouTube
```