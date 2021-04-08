<template>
  <div class="asideBox">
    <el-radio-group v-model="isCollapse" class="menu-toggle">
      <el-radio-button  :label="true" v-show="!isCollapse"><span class="el-icon-caret-left"></span></el-radio-button>
      <el-radio-button :label="false" v-show="isCollapse"><span class="el-icon-caret-right"></span></el-radio-button>
    </el-radio-group>
<!--    <div style="height: 40px;"></div>-->
      <el-aside style="width: 100%;">

          <el-menu router
                   background-color="#E9EEF3"
                   text-color="#333"
                   active-text-color="#fff"
                   :default-active="activeIndex"
                   :default-openeds="openSubMenu"
                   style="height: 100%;"
                   class="el-menu-vertical-demo"
                   @select="handleSelect"
                   :collapse="isCollapse"
                   unique-opened
          >
  <!--                 unique-opened>-->
            <!--background-color="rgb(247, 247, 247)"-->

            <el-submenu index="1">
              <template slot="title"><i class="el-icon-data-analysis"></i>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;系统管理</template>
              <el-menu-item index="/systemManagement/accountManagement">用户管理</el-menu-item>
              <el-menu-item index="/systemManagement/ticketNumberManagement">罚单编号管理</el-menu-item>
              <el-menu-item index="/systemManagement/redListManagement">红名单管理</el-menu-item>
            </el-submenu>
            <el-submenu index="2">
              <template slot="title"><i class="el-icon-view"></i>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;交警层级管理</template>
              <el-menu-item index="/policyHierarchyManagement/accessManagement">职级管理</el-menu-item>
              <el-menu-item index="/policyHierarchyManagement/hierarchyManagement">部门管理</el-menu-item>
            </el-submenu>
           <!-- <el-submenu index="3">  &lt;!&ndash;我&ndash;&gt;
              <template slot="title"><i class="el-icon-finished"></i>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;违法认定</template>
              <el-menu-item index="/nonMotor/unCheckPhotoNonMotor">非机动车</el-menu-item>
              <el-menu-item index="/pedestrian/unCheckPhoto">行人</el-menu-item>
            </el-submenu>-->
            <el-submenu index="4">
              <template slot="title"><i class="el-icon-folder-opened"></i>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;罚单管理</template>
              <el-submenu index="4-1" class="menu-level-2">
                <template slot="title"><i class="el-icon-document-checked"></i>&nbsp;&nbsp;罚单生成</template>
                <el-menu-item index="/ticketManagement/ticketGenerate/pede">行人</el-menu-item>
                <el-menu-item index="/ticketManagement/ticketGenerate/nonm">非机动车</el-menu-item>
              </el-submenu>
              <el-submenu index="4-2" class="menu-level-2">
                <template slot="title"><i class="el-icon-chat-dot-square"></i>&nbsp;&nbsp;短信通知</template>
                <el-menu-item index="/ticketManagement/messageManagement/pede">行人</el-menu-item>
                <el-menu-item index="/ticketManagement/messageManagement/nonm">非机动车</el-menu-item>
              </el-submenu>
              <el-submenu index="4-3" class="menu-level-2">
                <template slot="title"><i class="el-icon-position"></i>&nbsp;&nbsp;路口屏生成</template>
                <el-menu-item index="/ticketManagement/postScreenManagement/pede">行人</el-menu-item>
                <el-menu-item index="/ticketManagement/postScreenManagement/nonm">非机动车</el-menu-item>
              </el-submenu>
            </el-submenu>
            <el-submenu index="5"><!--我-->
              <template slot="title"><i class="el-icon-document"></i>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;违法档案管理</template>
              <el-menu-item index="/illegalFileManagement/illegalRecords">违法记录统计与分析</el-menu-item>
<!--              <el-menu-item index="/illegalFileManagement/recordsGraph">统计与分析图表</el-menu-item>-->
            </el-submenu>
          <!--  <el-submenu index="6">
              <template slot="title"><i class="el-icon-star-off"></i>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;字典维护</template>
              <el-menu-item index="/dictionaryManagement/roadDict">路段字典</el-menu-item>
              <el-menu-item index="/dictionaryManagement/illegalDict">违法字典</el-menu-item>
            </el-submenu>-->
            <el-submenu index="7">
              <template slot="title"><i
                class="el-icon-s-data"></i>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;交通违法数据可视化&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</template>
              <el-menu-item index="/visualization">可视化显示</el-menu-item>
            </el-submenu>
          </el-menu>
    </el-aside>

  </div>
</template>

<script>
import {mapState} from 'vuex'// 引用action和state映射函数
export default {
  name: 'Aside',
  data () {
    return {
      isCollapse: false,
      openSubMenu: []
    }
  },
  computed: {
    ...mapState({
      activeIndex: state => state.currentIndex// 获取当前激活菜单
    })
  },
  methods: {
    handleSelect: function (key, keyPath) {
      console.log(key)
      console.log(keyPath)
      // this.activeIndex = key
      // this.openSubMenu = keyPath
      // console.log(key)
      // this.activeIndex = key
      // this.setCurrentIndex(key)
      // let temp = []
      // for (let x of keyPath) {
      //   temp.push(x)
      // }
      // this.openSubMenu = temp
      // console.log(temp)
    }
    /* ...mapActions({
      setCurrentIndex: 'setCurrentIndex'
    }) */
  }
}
</script>

<style scoped>
  .el-menu-vertical-demo{
    text-align: left;

  }
  .asideBox{
    position: relative;
    padding-top: 20px;
    font-weight: bold;
    background-color: #E9EEF3;
    /*overflow: hidden;*/
    /*min-width: 60px;*/
    /*background-image: url("../../static/image/left_inner_bg_2.png");*/
  }
.el-submenu__title i {
  font-weight: bold;
  /*color:#FF7123;*/
}
  .el-aside {
    overflow: hidden;
    /*background-image: url("../../static/image/left_inner_bg_2.png");*/
  }
  .el-menu-item.is-active {
    background-color: #175FAA !important;
    /*background-color: #bae7ff !important;*/
    font-weight: bolder;
  }
  .el-submenu [class^="el-icon-"] {
    margin-right:8px;
  }
  /*.el-menu-item{*/
    /*background-color:#003a8c  !important;*/
    /*background-color: #002766 !important;*/
  /*}*/
  .menu-toggle{
    position: absolute;
    left: 100%;
    top: 225px;
    z-index: 20;
  }
</style>
<style>
  .el-submenu__title{
    font-size: 16px;
    /*padding-left: 21px !important;*/
  }
  .el-menu-item{
    font-size: 16px;
    height: 50px !important;
    line-height: 50px !important;
  }
  .el-menu-vertical-demo:not(.el-menu--collapse) {
    min-height: 400px;
  }
  .el-radio-button__orig-radio:checked+.el-radio-button__inner {
    color: #FFF;
    background-color: #175FAA;
    border-color: #175FAA;
    -webkit-box-shadow: -1px 0 0 0 #175FAA;
    box-shadow: -1px 0 0 0 #175FAA;
  }
  .el-radio-button__inner:hover{
    color: #175FAA;
  }
  .el-radio-button__inner{
    padding: 12px 0px !important;
    border-radius: 0 4px 4px 0 !important;
  }
  .el-menu--collapse .el-submenu {
    padding-left: 3px !important;
  }

  /*.el-submenu.menu-level-2{*/
  /*  padding-left: 20px !important;*/
  /*}*/
  /*.el-submenu.menu-level-2>ul{*/
  /*  padding-left: 38px;*/
  /*}*/
  /*!*一级菜单打开*!*/
  /*.el-submenu.is-opened{*/
  /*  background-color: #096dd9;*/
  /*}*/
</style>
