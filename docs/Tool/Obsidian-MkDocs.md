> [!info]+ obsidian图片路径问题
> obsidian中的图片路径是传统的markdown不一样，使用MkDocs部署的时候有显示问题，用了这个[mkdocs-obsidian-bridge插件](https://github.com/GooRoo/mkdocs-obsidian-bridge)完美解决，讨论可见[这里](https://github.com/GooRoo/mkdocs-obsidian-bridge/issues/17#issuecomment-2369535324)
> 
> ```shell
> pip install mkdocs-obsidian-bridge
> ```
> 
> 在`mkdocs.yml`里面修改
> 	```yaml
> 	plugins:
> 	   -obsidian-bridge
> 	markdown_extensions:
> 	  - obsidian_media_mkdocs  # this thing embeds audio/video/YouTube
> 	```


> [!info] 标题
> ```
> plugins:
>   - obsidian-bridge
> 
> markdown_extensions:
>   - obsidian_media_mkdocs
> ```

## debug记录

1. 使用github action部署报错
```
INFO - Copying '/home/runner/work/LearnAI-with-Ray/LearnAI-with-Ray/site' to 'gh-pages' branch and pushing to GitHub.  
remote: Permission to Ray-Lei-17/LearnAI-with-Ray.git denied to github-actions[bot].  
fatal: unable to access '[https://github.com/Ray-Lei-17/LearnAI-with-Ray/](https://github.com/Ray-Lei-17/LearnAI-with-Ray/)': The requested URL returned error: 403
```
- Go to your repository on GitHub.
- Navigate to **Settings** > **Actions** > **General**.
- Scroll down to the **Workflow permissions** section.
- Select **Read and write permissions**.
- Click **Save**.

2. 本地部署运行的很好，但是线上就是访问的模版不对，子目录也访问不到
- 忘记加上site url，发布的branch错了

3. Table of Content不对，内容格式出错
- 发现是因为mkdocs不能识别内容中的一级标题，把一级标题改成二级标题全部正常了

2. 公式渲染异常
- 参考这篇文章[mkdocs支持Latex风格的公式](https://seekstar.github.io/2024/03/21/mkdocs%E6%94%AF%E6%8C%81latex%E9%A3%8E%E6%A0%BC%E7%9A%84%E5%85%AC%E5%BC%8F/)
- 添加以下内容
```
markdown_extensions:
  - pymdownx.arithmatex

extra_javascript:
  - https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js?config=TeX-AMS-MML_HTMLorMML
```


>[!info]+ 标题折叠


>[!info]+ callout内代码行间距过大


>[!info]+ callout图标库



