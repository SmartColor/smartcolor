<!--Created By Gao HaoRan 2019.5.5-->
<template>
  <el-dropdown>
              <span class="el-dropdown-link">
               <i class="el-icon-third-ziyuan23" style="margin-right: 15px"></i>
               {{userName}}
              </span>
    <el-dropdown-menu slot="dropdown">
      <el-dropdown-item @click.native="logout">注销</el-dropdown-item>
    </el-dropdown-menu>
  </el-dropdown>
</template>

<script>
import {mapActions, mapState} from 'vuex'

export default {
  name: 'BriefInfo',
  computed: {
    ...mapState({
      userName: state => state.user.name,
      userPhotoPath: state => state.user.userPhotoPath
    })
  },
  methods: {
    // 注销用户
    logout: function () {
      this.logoutAction()
      this.$router.push({path: '/login'})
    },
    ...mapActions({
      logoutAction: 'logoutAction'
    })
  }
}
</script>

<style scoped>

</style>
