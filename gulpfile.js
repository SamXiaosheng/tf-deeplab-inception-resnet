const gulp = require('gulp')
const exec = require('child_process').exec

const Port = 9999
const Host = 'vincentchu@localhost'
const SrcDir = '/Users/vince/workspace/tf-deeplab-inception-resnet/'
const DestDir = '/home/vincentchu/workspace/tf-deeplab-inception-resnet'

gulp.task('sync', (cb) => {
  console.log('Syncing!')
  cmd = `rsync -avzr --delete --exclude='DATA/' --exclude='node_modules/' --exclude='.git/' --include='*.py' --include='*/' --exclude='*' --exclude='\.git/*' -e "ssh -p ${Port}" ${SrcDir} ${Host}:${DestDir}`
  exec(cmd, (err, stdout, stderr) => {
    console.log(stdout)
    cb()
  })
})

gulp.task('watch', (cb) => {
  gulp.watch('**/*.py', { debounceDelay: 10 }, ['sync'])
})

gulp.task('default', ['watch'])
