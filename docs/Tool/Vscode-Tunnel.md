# VS Code Tunnel
由于使用学校的VPN经常出现非常不稳定的情况，从同学那里知道可以使用vscode的tunnel，记录一下我的使用历程
[中文官方教程](https://vscode.github.net.cn/docs/remote/tunnels)
[英文官方教程](https://code.visualstudio.com/docs/remote/tunnels)
我的目标是不通过vpn直接从我本地电脑的vscode连接到远程服务器使用，这个服务实际上是使用微软托管的云服务（Microsoft-managed cloud service）来建立隧道链接
# 在远程机器安装tunnel
首先要下载code工具，我在服务器上使用中文官方提供的`vscode.github.net.cn`网址下载不下载，使用下面的英文版网址可以下载下来，或者可以在[下载网页](https://vscode.github.net.cn/#alt-downloads)下载命令行界面手动传上服务器
```
curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz
```
进行解压
```
tar -xf vscode_cli.tar.gz
```
解压之后得到一个code可执行程序，在该程序目录下执行命令
```
./code tunnel
```
然后会要求你登录github/微软账户，登录之后输入设备的编码，然后等待隧道生成，命名，此时在浏览器打开指定网址就可以进行编程，非常方便
然后在vscode的扩展商店上下载`Remote - Tunnels`插件，然后登录你的账户之后，在远程SSH的同一位置会出现登记在同一账号下的设备，直接打开就可以进行连接，不需要输入账号密码等，非常的方便
此时若还是执行着`./code tunnel`，还只是单次的服务运行，可以`Ctrl+C`退出单次运行，我们接下来在服务器上运行
```
./code tunnel service install
```
执行完之后，就可以非常方便的进行连接了。