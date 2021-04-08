<!--Created By Gao HaoRan 2019.4.28-->
<template>
  <!--  model:表单数据对象-->
  <!--  rules:表单验证规则-->
  <!--  label-position:标签对齐方式-->
  <div class="bgBox">
    <el-row>
      <el-col :lg="{span:6,offset:9}" :md="{span: 10,offset:7}">
        <el-form :model="loginForm" :rules="loginRules" class="demo-ruleForm login-container" label-position="left"
                 label-width="0px"
                 ref="loginForm">
          <h2 style="color: #409EFF">ColorBlend</h2>
          <!--    一定要注意！Form-Item 的 prop 属性设置为需校验的字段名-->
          <el-form-item class="item" prop="account">

            <!--  prefix-icon:输入框前部的图标-->
            <!--  v-model:数据绑定-->
            <el-input auto-complete="off" placeholder="Account" prefix-icon="" type="text"
                      v-model="loginForm.account">
            </el-input>
          </el-form-item>
          <el-form-item class="item" prop="pwd">
            <el-input auto-complete="off" placeholder="Password" prefix-icon="" type="password"
                      v-model="loginForm.pwd">
            </el-input>
          </el-form-item>
          <el-form-item class="submit">

            <!--      略过登录，仅限于纯前端测试情况-->
            <el-button :loading="logining" @click.native.prevent="ignoreLogin" class="button"
                       type="primary" plain size="middle">Sign in
            </el-button>
            <!--      实际登录-->
           <!-- <el-button :loading="logining" @click.native.prevent="login('loginForm')" class="button"
                       type="primary" plain size="small">Sign up
            </el-button>-->

          </el-form-item>
          <div class="signUp" style="color:#fff;">Don't have an account? <span
            style="color:#409EFF;font-weight: bold;cursor: pointer" >Sign up</span></div>
        </el-form>
      </el-col>
    </el-row>
  </div>

</template>

<script>
import md5 from 'js-md5' // 引用md5加密
import {mapActions, mapState} from 'vuex'// 引用action和state映射函数

export default {
  name: 'Login',
  data () {
    return {
      logining: false, // true为登陆中转态，loading图标出现
      loginForm: {// 登录表单
        account: '', // 账号
        pwd: ''// 密码
      },
      // 表单输入规则
      loginRules: {
        account: [
          {required: true, message: '请输入账号', trigger: 'blur'}
        ],
        pwd: [
          {required: true, message: '请输入密码', trigger: 'blur'}
        ]
      }
    }
  },
  computed: {
    /* mapState示例 */
    ...mapState({
      user: state => state.user// 获取用户
    })
    /****************/
  },
  methods: {
    // 提交登录表单事件
    /**
       * 用户登录
       * @param formName 表单名
       */
    login: function (formName) {
      // 验证表单合法性
      this.$refs[formName].validate((valid) => {
        if (valid) { // 表单合法
          this.logining = true// 显示‘加载中’图标

          // 发送登录请求
          return this.$axios.post('/login', {
            account: this.loginForm.account,
            pwd: this.encrypt(this.loginForm.pwd) // 密码加密
          })
            // 请求成功响应
            .then((res) => {
              this.logining = false// 取消‘加载中’图标

              let status = res.data.status
              let data = res.data.data
              let message = res.data.message
              // console.log(status)
              // console.log(data)
              // 登录成功
              if (status === 'success') {
                this.loginAction(data)// 登录数据写入
                // 检查token,这玩意儿应该直接push就行，错误检查交由beforeRouter
                if (this.user.token !== '') {
                  this.redirectByAccess()
                } else {
                  this.$router.replace('/login')// 退回登录页
                }

                this.$message.success('登录成功')
              } else if (data.status === 'fail') { // 登陆失败
                this.$message.error(message)
                this.$router.replace('/login')// 退回登录页
              }
            })
          // 请求出错
            .catch((error) => {
              this.logining = false
              console.log(error)
            })
        } else { // 输入不合法情况
          this.$message.error('账号密码输入格式不正确！')
          return false
        }
      })
    },
    /**
       * 加密算法
       * @param password 密码
       */
    encrypt (password) {
      return md5(password)
    },
    // 略过登录
    /**
       * 略过登录,仅供纯前端测试时使用
       */
    ignoreLogin: function () {
      let data = {
        account: '111',
        name: '111',
        position: '111',
        token: '111',
        access: ['1', '2', '3', '4', '5', '6', '7', '8', '9'],
        userPhotoPath: '111',
        department: '111'
      }
      this.loginAction(data)
      if (this.user.token !== '') {
        this.$router.push({name: 'Website'})// 前往Home页面
        // 根据权限自动跳转
        // this.redirectByAccess()
      } else {
        this.$router.replace('/login')// 退回登录页
      }
      // this.$router.replace('/home')
    },
    /**
       * 根据登录用户的权限自动跳转到合适页面
       * */
    redirectByAccess () {
      let minAccess = this.user.access[0]
      let to = {}
      let path = ''
      switch (minAccess) {
        case '1':
          // 违法认定 1
          to = {name: 'UnCheckPhotoNonMotor'}
          path = '/nonMotor/unCheckPhotoNonMotor'
          break
        case '2':
          // 罚单管理2
          to = {name: 'PedeNotGenIllegalInfo'}
          path = '/ticketManagement/ticketGenerate/pede'
          break
        case '3':
          to = {name: 'TicketNumberManagement'}
          path = '/systemManagement/ticketNumberManagement'
          // 罚单编号管理3
          break
        case '4':
          to = {name: 'AccountManagement'}
          path = '/systemManagement/accountManagement'
          // 用户管理4
          break
        case '5':
          to = {name: 'RedListManagement'}
          path = '/systemManagement/redListManagement'
          // 红名单管理5
          break
        case '6':
          to = {name: 'DictionaryManagement'}
          path = '/dictionaryManagement/roadDict'
          // 字典维护6
          break
        case '7':
          to = {name: 'PolicyHierarchyManagement'}
          path = '/policyHierarchyManagement/hierarchyManagement'
          // 交警层级管理 7
          break
        case '8':
          to = {name: 'IllegalFileManagement'}
          path = '/illegalFileManagement/illegalRecords'
          // 违规档案管理8
          break
        case '9':
          to = {name: 'Visualization'}
          path = '/visualization'
          // 交通违法数据可视化9
          break
        default:
          to = {name: 'UnCheckPhoto'}
          path = '/nonMotor/unCheckPhotoNonMotor'
          break
      }
      this.$router.push(to)// 前往首页面
      this.setCurrentIndex(path)
    },
    /**
       * action映射函数
       */
    ...mapActions({
      loginAction: 'loginAction', // 登录事件
      setCurrentIndex: 'setCurrentIndex'
    })
  }
}

</script>
<style scoped>
  html, body {
    margin: 0;
    width: 100%;
    height: 100%;
  }

  .bgBox {
    background: url("../../../static/image/背景3-2.png") no-repeat;
    background-size: cover;
    height: 100%;
  }

  .login-container {
    -webkit-border-radius: 5px;
    border-radius: 5px;
    -moz-border-radius: 5px;
    background-clip: padding-box;
    margin: 200px auto;
    width: 400px;
    padding: 15px 50px;
    background: rgba(0, 0, 0, .2);
    /*border: 1px solid #eaeaea;*/
    /*box-shadow: 0 0 25px #cac6c6;*/
    box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);

  }

  .title {
    margin: 0px auto 40px auto;
    text-align: center;
    color: #505458;
  }

  .submit {
    margin-top: 50px;
  }

  .button {
    font-size: large;
  }

  .remember {
    margin: 0px 0px 35px 0px;
  }
</style>
