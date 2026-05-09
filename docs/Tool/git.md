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
然后在github不同的账号设置下添加对应的ssh key即可，可用如下命令
```
git remote set-url origin git@github-user-A.com:github-user-A/your-repository.git
```

## 文件本地已删除，在git批量删除

```
git rm --cached $(git ls-files --deleted)
```

如果文件路径里有空格的话就用下面这个

```
git ls-files --deleted -z | xargs -0 git rm --cached
```

## git worktree

直接用分支名：

```bash
git worktree add ../新目录名 已有分支名
```

比如你有一个 `new-branch` 分支（且当前不在这个分支上）：

```bash
git worktree add ../新目录名 new-branch
```

就这一条命令，不需要 `-b`，因为分支已经存在，不用新建。但是注意如果是原创有该分支，而本地没有该分支，一样需要新建分支，使用带`-b`带命令。
```bash
git worktree add ../新目录名 -b new-branch origin/new-branch
```

---

**`-b` 和不加 `-b` 的区别：**

| 写法                                | 含义       |
| --------------------------------- | -------- |
| `git worktree add ../dir -b 新分支名` | 新建分支再检出  |
| `git worktree add ../dir 已有分支名`   | 直接检出已有分支 |
## git fetch

`fetch` 只是把远程的信息下载到 `.git` 里，**不会动你工作目录的任何文件**。

对比一下几个常见命令：

|命令|会改变工作目录文件吗|
|---|---|
|`git fetch`|❌ 不会|
|`git pull`|✅ 会（fetch + merge）|
|`git checkout`|✅ 会|
|`git merge`|✅ 会|

`fetch` 是最安全的操作，随时可以跑。