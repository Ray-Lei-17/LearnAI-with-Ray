document.addEventListener("DOMContentLoaded", function() {
    // 监听 H1 到 H6 的所有标题
    const headers = document.querySelectorAll("article h1, article h2, article h3, article h4, article h5, article h6");

    const isHeader = (el) => /^H[1-6]$/.test(el.tagName);
    const getHeaderLevel = (el) => parseInt(el.tagName.substring(1), 10);

    const setCollapsed = (header, collapsed) => {
        header.dataset.collapsed = collapsed ? "true" : "false";
        const icon = header.querySelector(".collapse-icon");
        if (icon) {
            icon.style.transform = collapsed ? "rotate(-90deg)" : "";
        }
    };

    const toggleSection = (header) => {
        const currentLevel = getHeaderLevel(header);
        const shouldCollapse = header.dataset.collapsed !== "true";
        let next = header.nextElementSibling;

        // 遍历逻辑：隐藏直到遇到【同级或更高级别】的标题
        while (next) {
            if (isHeader(next)) {
                const nextLevel = getHeaderLevel(next);
                if (nextLevel <= currentLevel) break; // 遇到同级或长辈标题，停止
            }

            // 执行隐藏或显示
            next.style.display = shouldCollapse ? "none" : "";
            next = next.nextElementSibling;
        }

        setCollapsed(header, shouldCollapse);
    };

    const expandSection = (header) => {
        if (!header || header.dataset.collapsed !== "true") return;
        toggleSection(header);
    };

    const getAncestorHeaders = (targetHeader) => {
        const ancestors = [];
        let current = targetHeader;
        let currentLevel = getHeaderLevel(current);

        while (true) {
            let prev = current.previousElementSibling;
            while (prev) {
                if (isHeader(prev)) {
                    const prevLevel = getHeaderLevel(prev);
                    if (prevLevel < currentLevel) {
                        ancestors.push(prev);
                        current = prev;
                        currentLevel = prevLevel;
                        break;
                    }
                }
                prev = prev.previousElementSibling;
            }
            if (!prev) break;
        }

        return ancestors;
    };

    const flashHeader = (header) => {
        if (!header) return;
        header.classList.remove("flash-target");
        // 触发重排以重启动画
        void header.offsetWidth;
        header.classList.add("flash-target");
        header.addEventListener(
            "animationend",
            () => header.classList.remove("flash-target"),
            { once: true }
        );
    };

    const revealHeaderFromHash = (hash) => {
        if (!hash || hash.length < 2) return;
        const targetId = decodeURIComponent(hash.substring(1));
        const target = document.getElementById(targetId);
        if (!target) return;

        const header =
            target.closest("h1, h2, h3, h4, h5, h6") ||
            (isHeader(target) ? target : null);
        if (!header) return;

        getAncestorHeaders(header).forEach(expandSection);
        expandSection(header);
        flashHeader(header);
    };

    headers.forEach((header) => {
        // 设置鼠标样式并添加交互标识
        header.style.cursor = "pointer";
        header.classList.add("collapsible-header");
        header.dataset.collapsed = "false";

        // 在标题前插入一个控制符号（你可以用更现代的 SVG）
        const icon = document.createElement("span");
        icon.className = "collapse-icon";
        icon.innerHTML = "▾"; // 默认展开状态的符号
        icon.style.marginRight = "10px";
        icon.style.transition = "transform 0.2s";
        icon.style.display = "inline-block";
        header.prepend(icon);

        header.addEventListener("click", function(event) {
            if (event.target.closest("a")) return;
            toggleSection(this);
        });
    });

    document.addEventListener("click", function(event) {
        const link = event.target.closest('a[href^="#"]');
        if (!link) return;
        const href = link.getAttribute("href");
        if (!href || href === "#") return;
        revealHeaderFromHash(href);
    });

    window.addEventListener("hashchange", function() {
        revealHeaderFromHash(window.location.hash);
    });

    revealHeaderFromHash(window.location.hash);
});
