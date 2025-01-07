# Mac开启HiDPI
Mac默认对4K显示器有HiDPI支持，但是对2K显示器没有相关支持，见[Apple社区](https://discussionschinese.apple.com/thread/252666081?sortBy=rank)

可以使用这个[仓库](https://github.com/xzhih/one-key-hidpi)中的代码来修改开启HiDPI，非常方便，需要外网访问。20240107，执行如下
```shell
bash -c "$(curl -fsSL https://raw.githubusercontent.com/xzhih/one-key-hidpi/master/hidpi.sh)"
```
![[../Images/Pasted image 20250107173405.png]]

开启完之后，分辨率后面会显示HiDPI，这时选择相应分辨率，实际分辨率不会下降，只起到缩放效果。
![[../Images/Pasted image 20250107173009.png]]
