<template>
    <div id="app">
        <router-view/>
    </div>
</template>

<script>
export default {
  data () {
    return {}
  },
  components: {},
  name: 'App',
  methods: {}
}
</script>

<style>
  html{
    height: 100%;
  }
  body{
    height: 100%;
    margin: 0 !important;
  }
    #app {
        /*font-family: 'Avenir', Helvetica, Arial, sans-serif;*/

        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        text-align: center;
        color: #2c3e50;
        height: 100%;
        font-size: 18px;
      /*font-family: "Helvetica Neue","Microsoft YaHei", Helvetica, Arial, sans-serif;;*/
        font: normal 18px/1.6 'Open Sans', helvetica, arial, sans-serif;
        color: #8e8e8e;
    }

    .register:only-child {
        min-width: 350px;
    }
  .el-container{
    /*min-height: 100% !important;*/
  }

  /*页面公共样开始*/

    /*所有表格公共样式*/
    #table{
      /*margin-top: 15px;*/
      padding: 25px 20px 25px 20px;
      background-color: #ffffff;
      margin-bottom: 20px;
    }
  .el-dialog__body{
    padding: 10px 20px !important;
  }
  ul{
    list-style: none;
    padding: 0px;
  }
  .el-table .warning-row {
    background: oldlace !important;
  }

  .el-table .success-row {
    background: #f0f9eb !important;
  }
    /*页面公共样式结束*/
  input.el-range-input{
    background-color: transparent!important;
  }
  .el-input__inner .el-range-separator {
    color: #DCDFE6!important ;
  }
</style>
