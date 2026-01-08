document.addEventListener("DOMContentLoaded", function() {
    // 监听 H1 到 H6 的所有标题
    const headers = document.querySelectorAll("article h1, article h2, article h3, article h4, article h5, article h6");
    
    headers.forEach(header => {
        // 设置鼠标样式并添加交互标识
        header.style.cursor = "pointer";
        header.classList.add("collapsible-header");
        
        // 在标题前插入一个控制符号（你可以用更现代的 SVG）
        const icon = document.createElement("span");
        icon.innerHTML = "▾"; // 默认展开状态的符号
        icon.style.marginRight = "10px";
        icon.style.transition = "transform 0.2s";
        icon.style.display = "inline-block";
        header.prepend(icon);

        header.addEventListener("click", function() {
            const currentLevel = parseInt(this.tagName.substring(1));
            let next = this.nextElementSibling;
            
            // 切换箭头方向
            const isHidden = icon.style.transform === "rotate(-90deg)";
            icon.style.transform = isHidden ? "" : "rotate(-90deg)";
            
            // 遍历逻辑：隐藏直到遇到【同级或更高级别】的标题
            while (next) {
                const nextTagName = next.tagName;
                if (/^H[1-6]$/.test(nextTagName)) {
                    const nextLevel = parseInt(nextTagName.substring(1));
                    if (nextLevel <= currentLevel) break; // 遇到同级或长辈标题，停止
                }
                
                // 执行隐藏或显示
                next.style.display = isHidden ? "" : "none";
                next = next.nextElementSibling;
            }
        });
    });
});