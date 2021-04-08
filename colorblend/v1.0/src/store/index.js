import Vue from 'vue'
import Vuex from 'vuex'
import axios from 'axios'
import createPersistedState from 'vuex-persistedstate'// vuex持久化

// 读取mutation命名，好处是有自动补全*_*
import {
  SET_NAME, DEL_NAME,
  SET_TOKEN, DEL_TOKEN,
  SET_POSITION, DEL_POSITION,
  SET_ACCOUNT, DEL_ACCOUNT,
  SET_ACCESS, DEL_ACCESS,
  SET_DEPARTMENT, DEL_DEPARTMENT,
  SET_USERPHOTOPATH, DEL_USERPHOTOPATH,
  SET_ROADLIST,
  SET_DEPTLIST,
  SET_ACCESSLIST,
  SET_CURRENTINDEX
} from './mutation-types'

Vue.use(Vuex)

// const debug = process.env.NODE_ENV !== 'production'

export default new Vuex.Store({
  state: {
    user: {
      name: '', // 用户名
      token: '', // 会话令牌
      position: '', // 用户职称
      account: '', // 用户ID
      access: [], // 用户权限
      department: '', // 部门
      userPhotoPath: ''// 头像
    },
    // 一些通用的数据
    commonData: {
      roadList: [], // 路段
      deptList: [], // 部门
      accessList: []// 权限
    },
    currentIndex: '/',
    /* 新添加的 */
    themecolor: '409EFF'// 默认
  },
  // 页面数据持久化，state缓存进sessionstorage
  plugins: [createPersistedState({
    storage: window.sessionStorage
  })],
  mutations: {
    // 用户名
    [SET_NAME] (state, name) {
      state.user.name = name
    },
    [DEL_NAME] (state) {
      state.user.name = ''
    },
    // 会话令牌
    [SET_TOKEN] (state, token) {
      state.user.token = token
    },
    [DEL_TOKEN] (state) {
      state.user.token = ''
    },
    // 用户角色权限
    [SET_POSITION] (state, position) {
      state.user.position = position
    },
    [DEL_POSITION] (state) {
      state.user.position = ''
    },
    // 用户ID
    [SET_ACCOUNT] (state, account) {
      state.user.account = account
    },
    [DEL_ACCOUNT] (state) {
      state.user.account = ''
    },
    // 用户权限
    [SET_ACCESS] (state, access) {
      state.user.access = access
    },
    [DEL_ACCESS] (state) {
      state.user.access = ''
    },
    // 用户部门
    [SET_DEPARTMENT] (state, department) {
      state.user.department = department
    },
    [DEL_DEPARTMENT] (state) {
      state.user.department = ''
    },
    // 用户头像
    [SET_USERPHOTOPATH] (state, userPhotoPath) {
      state.user.userPhotoPath = userPhotoPath
    },
    [DEL_USERPHOTOPATH] (state) {
      state.user.userPhotoPath = ''
    },
    // 设置可选路段
    [SET_ROADLIST] (state, roadlist) {
      state.commonData.roadList = roadlist
    },
    // 设置可选部门
    [SET_DEPTLIST] (state, deptlist) {
      state.commonData.deptList = deptlist
    },
    // 设置当前激活菜单
    [SET_CURRENTINDEX] (state, index) {
      state.currentIndex = index
    },
    // 更新主题颜色
    setThemeColor (state, curcolor) {
      this.state.themecolor = curcolor
    }
  },
  actions: {

    // 成功登录事件
    loginAction ({commit, state}, {account, name, position, token, access, userPhotoPath, department}) {
      commit(SET_NAME, name)
      commit(SET_ACCOUNT, account)
      commit(SET_POSITION, position)
      commit(SET_TOKEN, token)
      commit(SET_ACCESS, access)
      commit(SET_DEPARTMENT, department)
      commit(SET_USERPHOTOPATH, userPhotoPath)
    },
    // 注销事件
    logoutAction ({commit}) {
      commit(DEL_ACCOUNT)
      commit(DEL_POSITION)
      commit(DEL_TOKEN)
      commit(DEL_NAME)
      commit(DEL_ACCESS)
      commit(DEL_DEPARTMENT)
      commit(DEL_USERPHOTOPATH)
    },

    /* 新加的 */
    /**
     * @Description
     * @author Liu Huiyang
     * @date 2019/5/10
     */

    /* * **********************************************行人开始***************************** */
    /* 行人获取初始列表 */
    getIegPedestrianList (store, obj) {
      return axios.post('/IegPedestrian/getIegPedestrianList', {
        place: obj.place,
        state: obj.state, /* 0代表未审核模块  1代表未确认模块   2代表已确认模块   3代表无法确认模块 */
        order: obj.order, /* 0降序 1升序 */
        threshold: obj.threshold, /* 阈值 不加 % */
        index: obj.index, /* 页数 */
        date: JSON.stringify(obj.date),
        keyword: obj.keyword ? obj.keyword : ''
      })
    },
    /* 审核 */
    check (store, obj) {
      return axios.post('/IegPedestrian/getIegPedestrian', {
        pedeId: obj.pedeId

      })
    },
    /* 稍后处理、无法确认 */
    unDeal (store, obj) {
      return axios.post('/IegPedestrian/updatePedestrianIegState', {
        pedeId: obj.pedeId,
        state: obj.state,
        candidatePath: obj.candidatePath

      })
    },
    /* 确认 */
    conform (store, obj) {
      return axios.post('/IegPedestrian/confirmPedestrianIeg', {
        pedeId: obj.pedeId,
        dsr: obj.dsr,
        name: obj.name,
        file: obj.file

      })
    },
    /* 批量审核初始数据获取 */
    getcount (store, obj) {
      return axios.post('/IegPedestrian/getBatchPedestrianIegCount', {
        threshold: obj.threshold,
        date: JSON.stringify(obj.date)

      })
    },
    /* 批量审核无法确认 */
    checkLUnconfirm (store, obj) {
      return axios.post('/IegPedestrian/batchUpdatePedestrianIegState', {
        threshold: obj.threshold,
        date: JSON.stringify(obj.date),
        state: obj.state

      })
    },

    /* * **********************************************行人结束***************************** */

    /* * **********************************************非机动车开始***************************** */
    getIegNonmList (store, obj) {
      return axios.post('/IegNonm/getIegNonmList', {
        place: obj.place,
        state: obj.state, /* 0代表未审核模块  1代表未确认模块   2代表已确认模块   3代表无法确认模块 */
        order: obj.order, /* 0降序 1升序 */
        threshold: obj.threshold, /* 阈值 不加 % */
        index: obj.index, /* 页数 */
        date: JSON.stringify(obj.date),
        keyword: obj.keyword ? obj.keyword : ''
      })
    },
    /* 审核 */
    checkNonmotor (store, obj) {
      return axios.post('/IegNonm/getIegNonm', {
        nonmId: obj.nonmId

      })
    },
    /* 稍后处理、无法确认 */
    unDealNonmotor (store, obj) {
      return axios.post('/IegNonm/updateNonmIegState', {
        nonmId: obj.nonmId,
        state: obj.state,
        candidatePath: obj.candidatePath,
        backPath: obj.backPath,
        hphmPath: obj.hphmPath

      })
    },

    /* 确认 */
    conformNonmotor (store, obj) {
      return axios.post('/IegNonm/confirmNonmIeg', {
        nonmId: obj.nonmId,
        dsr: obj.dsr,
        name: obj.name,
        candidatePath: obj.candidatePath,
        hphm: obj.hphm,
        backPath: obj.backPath,
        hphmPath: obj.hphmPath
      })
    },

    /* 批量审核初始数据获取 */
    getcountNonmotor (store, obj) {
      return axios.post('/IegNonm/getBatcheNonmIegCount', {
        threshold: obj.threshold,
        date: JSON.stringify(obj.date)

      })
    },

    /* 批量审核无法确认 */
    checkLUnconfirmNonmotor (store, obj) {
      return axios.post('/IegNonm/batchUpdateNonmIegState', {
        threshold: obj.threshold,
        date: JSON.stringify(obj.date),
        state: obj.state

      })
    },
    /* * **********************************************非机动车结束***************************** */

    /* ***************************************违规档案管理开始******************************************/
    /* 行人获取初始列表 */
    getIegInfoList (store, obj) {
      return axios.post('/IegInfoMng/getIegInfoList', {
        date: JSON.stringify(obj.date),
        place: obj.place,
        category: obj.category,
        top: obj.top,
        index: obj.index,
        times: obj.times
      })
    },

    /* ***************************************违规档案管理结束******************************************/

    /* ***************************************公共接口开始******************************************/
    /**
     * 设置部门所有权限列表
     * Gao HaoRan
     */
    setDeptAccess ({commit, state}) {
      axios.get('/Common/getDeptRootList')
        .then((res) => {
          let data = res.data.data// 响应数据
          let status = res.data.status// 响应状态
          // let message = res.data.message// 响应消息
          if (status === 'success') { // 获取数据成功
            // this.deptAccess = this.toStringArray(data.accessFunction)
            commit(SET_ACCESSLIST, data.accessFunction)
          }
          // else if (status === 'fail') { // 错误处理尝试外移
          //   this.$message.error(message)
          // }
        })
        .catch((error) => {
          console.log(error)
        })
    },
    /**
     * 设置下级部门编号列表
     * @returns {Promise<T | never>}
     *Gao HaoRan
     */
    setDepartments ({commit, state}) {
      axios.get('/Common/getDepList')
        .then((res) => {
          let data = res.data.data// 响应数据
          let status = res.data.status// 响应状态
          if (status === 'success') { // 获取数据成功
            let organizations = []
            for (let x of data.depList) {
              let tempItem = {
                value: x
              }
              organizations.push(tempItem)
            }
            commit(SET_DEPTLIST, organizations)
          }
          // else if (status === 'fail') { // 错误处理尝试外移
          //   this.$message.error(message)
          // }
        })
        .catch((error) => {
          console.log(error)
        })
    },
    /**
     * 获取当前用户管理路段
     * Gao HaoRan
     * */
    setRoadList ({commit, state}) {
      axios.get('/Common/getIntersectionList')
        .then((res) => {
          let data = res.data.data// 响应数据
          let status = res.data.status// 响应状态
          // let message = res.data.message// 响应消息
          if (status === 'success') { // 获取数据成功
            commit(SET_ROADLIST, data.roadList)
          }
          // else if (status === 'fail') { // 错误处理尝试外移
          //   this.$message.error(message)
          // }
        })
        .catch((error) => {
          console.log(error)
        })
    },
    /* ***************************************公共接口结束******************************************/
    /* ***************************************应用状态开始******************************************/
    /**
     * 设置当前激活菜单
     * Gao HaoRan
     * */
    setCurrentIndex ({commit, state}, index) {
      commit(SET_CURRENTINDEX, index)
    }
    /* ***************************************应用状态结束******************************************/
  }
})
