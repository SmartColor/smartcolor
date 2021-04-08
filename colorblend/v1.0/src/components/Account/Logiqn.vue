
<template>
  <div class="bgBox">
  </div>
</template>
<script>
import md5 from 'js-md5' // 引用md5加密
import {mapActions, mapState} from 'vuex'// 引用action和state映射函数

export default {
  name: 'Login',
  data () {
    return {
      logining: false,
      loginForm: {
        account: '',
        pwd: ''
      }
    }
  },
  computed: {
    /* mapState示例 */
    ...mapState({
      user: state => state.user// 获取用户
    })
  },
  methods: {
    login: function () { },
    ...mapActions({
      loginAction: 'loginAction', // 登录事件
      setCurrentIndex: 'setCurrentIndex'
    })
  }
}
</script>
<style>
</style>
