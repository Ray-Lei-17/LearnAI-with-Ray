## 如何通过不同账号提交git

如果有通过不同账号提交的需要，可以通过一下子的方式进行操作。主要思想是：通过改变git remote url的host，在ssh config中指定不同的IdentityFile去访问

假设有两个账号，用户名如下
- github-user-A
- github-user-B

如果原来没有ssh key（默认文件夹位置在`~\.ssh\`），需要运行下面两行命令产生不同的key，在`Enter file in which to save the key`后面取不同的名字就好。
```
ssh-keygen -t ed25519 -C "your comment A"
ssh-keygen -t ed25519 -C "your comment B"
```

假设你生成的两个key的公共密钥位置在
- `~\.ssh\id_ed25519_user_A.pub`
- `~\.ssh\id_ed25519_user_B.pub`

现在修改`~/.ssh/config`，添加下面的内容
```
Host github-user-A.com
  HostName github.com
  User git
  AddKeysToAgent yes
  IdentityFile ~\.ssh\id_ed25519_user_A
  IdentitiesOnly yes

Host github-user-B.com
  HostName github.com
  User git
  AddKeysToAgent yes
  IdentityFile ~\.ssh\id_ed25519_user_B
  IdentitiesOnly yes
```
然后下面修改你的git仓库remote url，比如原来是github-user-A的账户下的仓库
```
git@github.com:github-user-A/your-repository.git
```
修改为
```
git@github-user-A.com:github-user-A/your-repository.git
```
然后在github不同的账号设置下添加对应的ssh key即可

## 文件本地已删除，在git批量删除

```
git rm --cached $(git ls-files --deleted)
```

如果文件路径里有空格的话就用下面这个

```
git ls-files --deleted -z | xargs -0 git rm --cached
```