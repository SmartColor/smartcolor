// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from 'vue'
import App from './App'
import router from './router'
import store from './store'
import ElementUI from 'element-ui'
import 'element-ui/lib/theme-chalk/index.css'
import './assets/icon/iconfont.css'
import moment from 'moment'// 时间格式化
import axios from 'axios'// http封装
import qs from 'qs'// 数据格式化

Vue.use(ElementUI)

// 全局绑定，以后使用的时候直接this.$moment就好了
Object.defineProperty(Vue.prototype, '$moment', { value: moment })
Object.defineProperty(Vue.prototype, '$axios', { value: axios })

Vue.config.productionTip = false
Vue.config.devtools = true

// 两台在一楼实验室的后端机器ip地址
axios.defaults.baseURL = 'http://10.101.0.52:8989'// renzhen
// axios.defaults.baseURL = 'http://10.101.71.19:8989'// wangmengyuan
// 更改post请求头部：编码格式部分
axios.defaults.headers.post['Content-Type'] = 'application/x-www-form-urlencoded'
// 添加请求拦截器
axios.interceptors.request.use(function (config) { // 更改axios编码格式
  // console.log(config)
  if (config.method === 'post') {
    config.data = qs.stringify(config.data)
  }
  if (store.state.user.token) { // 加上token
    config.headers.common['Authentication-Token'] = store.state.user.token
  }
  // 在发送请求之前做些什么
  return config
}, function (error) {
  // 对请求错误做些什么
  return Promise.reject(error)
})

// 添加响应拦截器
axios.defaults.headers.common['Authentication-Token'] = store.state.user.token// 好像写重了，懒得改了
axios.interceptors.response.use(function (res) {
  let state = res.data.status
  // 添加全局的错误处理（暂定）
  if (state === 'fail') {
    let message = res.data.message
    ElementUI.Message.error(message)
  }
  // 对响应数据做点什么
  return res
}, function (error) {
  // 对响应错误做点什么
  return Promise.reject(error)
})
/* eslint-disable no-new */

/* 权限控制 */
router.beforeEach((to, from, next) => {
  // 用户登录状态判断
  if (to.matched.some(r => r.meta.requireAuth)) {
    // 这里的requireAuth为路由中定义的 meta:{requireAuth:true}，意思为：表示进入该路由需要登陆
    if (store.state.user.token) {

    } else {
      ElementUI.Message('请登录')
      next({
        path: '/login'
      })
      return
    }
  }

  // 路由权限限制判断
  if (to.matched.some(r => r.meta.access)) {
    if (store.state.user.access) { // 如果用户有权限数组
      let userAccess = store.state.user.access
      let routerAccess = []
      let routers = to.matched// 匹配的路由数组
      for (let x of routers) {
        if (x.meta.access) {
          routerAccess = routerAccess.concat(x.meta.access)// 路由权限限制累加
        }
        /* 通过，或者不通过提示 */
      }
      routerAccess = [...new Set(routerAccess)]// 去重
      for (let x of routerAccess) { // 检查权限
        if (!userAccess.includes(x)) { // 没有权限
          ElementUI.Message.warning('您没有访问该模块的权限！')
          next(false)// 路由终止
          return
        }
      }
      next()
    } else {
      ElementUI.Message.warning('您没有访问该模块的权限！')
      next(false)// 路由终止
      return
    }
  }
  if (to.matched.some(r => r.meta.isMenuItem)) { // 给所有的需要监听的路由都在router index.js中添加meta.isMenuItem属性 通过是否有meta.isMenuItem属性来监听to当前的路由，通过路由的path设置左侧导航栏选中的activeIndex
    let currentIndex = to.redirectedFrom ? to.redirectedFrom : to.path
    // console.log(currentIndex)
    store.state.currentIndex = currentIndex
    if (currentIndex === '/') {
      store.state.currentIndex = '/website'
    }
  }
  next()
})

new Vue({
  el: '#app',
  router,
  store,
  components: { App },
  template: '<App/>'
})
