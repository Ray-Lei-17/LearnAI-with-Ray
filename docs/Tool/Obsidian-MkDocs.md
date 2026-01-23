>[!success] mkdocs本地使用
>```shell
>mkdocs build
>mkdocs serve
>```
>主要是上面两个命令，build用于静态构建，如果没有修改一些比较底层的内容，一般serve就够了，如果担心有缓存问题就执行
>```shell
>mkdocs build --clean
>mkdocs serve --cleah
>```
>然后在浏览器进入本地网址即可看到预览效果，pipeline已经写好了，如果要上线部署，直接push即可，如果有pip install新的包，在workflow中进行对应的修改

>[!question]+  obsidian图片路径问题
> obsidian中的图片路径是传统的markdown不一样，使用MkDocs部署的时候有显示问题，用了这个[mkdocs-obsidian-bridge插件](https://github.com/GooRoo/mkdocs-obsidian-bridge)完美解决，讨论可见[这里](https://github.com/GooRoo/mkdocs-obsidian-bridge/issues/17#issuecomment-2369535324)
> 
> ```shell
> pip install mkdocs-obsidian-bridge
> ```
> 
> 在`mkdocs.yml`里面修改
> 
> ```yaml
> plugins:
>   - obsidian-bridge
> markdown_extensions:
>   - obsidian_media_mkdocs  # this thing embeds audio/video/YouTube
> ```
>
>图片路径之外还遇到图片尺寸的调整在 obsidian中显示正常，但是在mkdocs中却没有成功调整尺寸，最后发现是因为在尺寸前面有空格就没法正确识别
>
>错误示范
>```
>![[Pasted image 20260122161346.png | 300]]
>```
>正确显示
>```
>![[Pasted image 20260122161346.png |300]]
>```

>[!question]+  使用github action部署报错
> ```
> INFO - Copying '/home/runner/work/LearnAI-with-Ray/LearnAI-with-Ray/site' to 'gh-pages' branch and pushing to GitHub.  
> remote: Permission to Ray-Lei-17/LearnAI-with-Ray.git denied to github-actions[bot].  
> fatal: unable to access '[https://github.com/Ray-Lei-17/LearnAI-with-Ray/](https://github.com/Ray-Lei-17/LearnAI-with-Ray/)': The requested URL returned error: 403
> ```
>   
> - Go to your repository on GitHub.
> - Navigate to **Settings** > **Actions** > **General**.
> - Scroll down to the **Workflow permissions** section.
> - Select **Read and write permissions**.
> - Click **Save**.

>[!question]+ 本地部署运行的很好，但是线上就是访问的模版不对，子目录也访问不到
> - 忘记加上site url，发布的branch错了

>[!question]+  Table of Content不对，内容格式出错
> - 发现是因为mkdocs不能识别内容中的一级标题，把一级标题改成二级标题全部正常了
 
> [!question]+ 公式渲染异常
> - 参考这篇文章[mkdocs支持Latex风格的公式](https://seekstar.github.io/2024/03/21/mkdocs%E6%94%AF%E6%8C%81latex%E9%A3%8E%E6%A0%BC%E7%9A%84%E5%85%AC%E5%BC%8F/)
> - 添加以下内容
> ```yaml
> markdown_extensions:
>   - pymdownx.arithmatex
> 
> extra_javascript:
>   - https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js?config=TeX-AMS-MML_HTMLorMML
> ```


>[!success]+ 标题折叠
>mkdocs这种静态网页默认是没有类似obsidian这种标题折叠，但是我有一些写的太啰嗦了，我会希望有折叠功能缩短文章，所以让gpt帮我写了js和css，把下面这两个内容在`mkdocs.yml`里包括进去就能实现：
>
>- 标题折叠功能，带标题右侧小箭头及旋转功能
>- 折叠后点击右侧目录中被折叠的标题实现自动展开跳转
>- 跳转后标题背景短暂变色显示
>
> js文件内容：
> 
> ```java
> document.addEventListener("DOMContentLoaded", function() {
>     // 监听 H1 到 H6 的所有标题
 >    const headers = document.querySelectorAll("article h1, article h2, article h3, article h4, article h5, article h6");
>     const isHeader = (el) => /^H[1-6]$/.test(el.tagName);
>     const getHeaderLevel = (el) => parseInt(el.tagName.substring(1), 10);
> 
>     const setCollapsed = (header, collapsed) => {
>         header.dataset.collapsed = collapsed ? "true" : "false";
>         const icon = header.querySelector(".collapse-icon");
>         if (icon) {
>             icon.style.transform = collapsed ? "rotate(-90deg)" : "";
>         }
>     };
> 
>     const toggleSection = (header) => {
>         const currentLevel = getHeaderLevel(header);
>         const shouldCollapse = header.dataset.collapsed !== "true";
>         let next = header.nextElementSibling;
> 
>         // 遍历逻辑：隐藏直到遇到【同级或更高级别】的标题
>         while (next) {
>             if (isHeader(next)) {
>                 const nextLevel = getHeaderLevel(next);
>                 if (nextLevel <= currentLevel) break; // 遇到同级或长辈标题，停止
>             }
> 
>             // 执行隐藏或显示
>             next.style.display = shouldCollapse ? "none" : "";
>             next = next.nextElementSibling;
>         }
> 
>         setCollapsed(header, shouldCollapse);
>     };
> 
>     const expandSection = (header) => {
>         if (!header || header.dataset.collapsed !== "true") return;
>         toggleSection(header);
>     };
> 
>     const getAncestorHeaders = (targetHeader) => {
>         const ancestors = [];
>         let current = targetHeader;
>         let currentLevel = getHeaderLevel(current);
> 
>         while (true) {
>             let prev = current.previousElementSibling;
>             while (prev) {
>                 if (isHeader(prev)) {
>                     const prevLevel = getHeaderLevel(prev);
>                     if (prevLevel < currentLevel) {
>                         ancestors.push(prev);
>                         current = prev;
>                         currentLevel = prevLevel;
>                         break;
>                     }
>                 }
>                 prev = prev.previousElementSibling;
>             }
>             if (!prev) break;
>         }
> 
>         return ancestors;
>     };
> 
>     const flashHeader = (header) => {
>         if (!header) return;
>         header.classList.remove("flash-target");
>         // 触发重排以重启动画
>         void header.offsetWidth;
>         header.classList.add("flash-target");
>         header.addEventListener(
>             "animationend",
>             () => header.classList.remove("flash-target"),
>             { once: true }
>         );
>     };
> 
>     const revealHeaderFromHash = (hash) => {
>         if (!hash || hash.length < 2) return;
>         const targetId = decodeURIComponent(hash.substring(1));
>         const target = document.getElementById(targetId);
>         if (!target) return;
> 
>         const header =
>             target.closest("h1, h2, h3, h4, h5, h6") ||
>             (isHeader(target) ? target : null);
>         if (!header) return;
> 
>         getAncestorHeaders(header).forEach(expandSection);
>         expandSection(header);
>         flashHeader(header);
>     };
> 
>     headers.forEach((header) => {
>         // 设置鼠标样式并添加交互标识
>         header.style.cursor = "pointer";
>         header.classList.add("collapsible-header");
>         header.dataset.collapsed = "false";
> 
>         // 在标题前插入一个控制符号（你可以用更现代的 SVG）
>         const icon = document.createElement("span");
>         icon.className = "collapse-icon";
>         icon.innerHTML = "▾"; // 默认展开状态的符号
>         icon.style.marginRight = "10px";
>         icon.style.transition = "transform 0.2s";
>         icon.style.display = "inline-block";
>         header.prepend(icon);
> 
>         header.addEventListener("click", function(event) {
>             if (event.target.closest("a")) return;
>             toggleSection(this);
>         });
>     });
> 
>     document.addEventListener("click", function(event) {
>         const link = event.target.closest('a[href^="#"]');
>         if (!link) return;
>         const href = link.getAttribute("href");
>         if (!href || href === "#") return;
>         revealHeaderFromHash(href);
>     });
> 
>     window.addEventListener("hashchange", function() {
>         revealHeaderFromHash(window.location.hash);
>     });
> 
>     revealHeaderFromHash(window.location.hash);
> });
> ```
> 
> css文件内容：
> ```css
> /* 鼠标悬停在标题上时背景微亮 */
> .collapsible-header:hover {
>     background-color: rgba(0, 0, 0, 0.02);
>     border-radius: 4px;
> }
> 
> .collapsible-header.flash-target {
>     animation: header-flash 1.2s ease-out;
>     border-radius: 4px;
> }
> 
> @keyframes header-flash {
>     0% {
>         background-color: rgba(255, 214, 0, 0.35);
>     }
>     60% {
>         background-color: rgba(255, 214, 0, 0.2);
>     }
>     100% {
>         background-color: transparent;
>     }
> }
> 
> /* 隐藏时平滑过渡的提示 */
> .collapse-icon {
>     color: var(--md-typeset-color);
>     opacity: 0.5;
> }
> ```
> 把上面两个文件创建好，放在`docs`文件夹下的某个位置，然后在`mkdocs.yml`里把这两个文件包括进去即可
>```yaml
>extra_javascript:
>   - assets/javascripts/collapse.js（换成自己的路径，这是docs内的路径）
>extra_css:
>   - assets/stylesheets/extra.css （换成自己的路径，这是docs内的路径）
>```


>[!question]+ callout内若出现hyphen sign(-)，会出现多余空行
>使用>[!info]在obsidian中显示正常，但是在mkdocs中，- 行前面出现多余空行，下面是我使用插件，也可以作为一个示例
>
>```yaml
> plugins:
>   - callouts
>```
>
>上面这段代码在obsidian看起来是这样的
>
>![[Pasted image 20260112190828.png|100]]
>
>在mkdocs里看起来如下
> 
>![[Pasted image 20260112191357.png|100]]
>
>会宽很多，下面的代码又会紧连在一起，看起来非常奇怪，排查了很久，废了很大劲，终于发现是由于mkdocs的`callouts` plugins导致的问题，可以通过控制其中的`breakless_lists`属性解决
>像下面这样即可
>
>```yaml
>plugins:
>  - callouts:
>      breakless_lists: false
>```


>[!warning]+ obsidian中admonition块渲染
> obsidian中默认是不支持mkdocs中!!!和???类似的写法渲染，如果想直接按照mkdocs写法在obsidian显示出来，目前我在插件市场没有看到一个合适的插件，所以用codex写了一个可供使用：
> 
>- [Ray-Lei-17/mkdocs-material-admonitions](https://github.com/Ray-Lei-17/mkdocs-material-admonitions)
>
>支持的类型：
>
>- note
>- info
>- tip
>- warning
>- important
>- caution
>- danger
>- bug
>- example
>- quote
>- failure
>- success
>- question


> [!success] Obsidian 关系图谱
> 我会使用关系图谱看是否存在不必要的多余的图片，这个时候需要去除site文件夹对我的影响，所我会使用如下筛选条件
> ```
> path:docs (file:.md OR file:.png)
> ```