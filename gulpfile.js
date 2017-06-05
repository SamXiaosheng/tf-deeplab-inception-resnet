const gulp = require('gulp')
const exec = require('child_process').exec


const Port = 9999
const Host = 'vincentchu@localhost'
const SrcDir = ''
const DestDir = '/home/vincentchu/workspace/tf-deeplab-inception-resnet'

gulp.task('sync', (cb) => {
  console.log('Syncing!')
  cmd = `rsync -avzr --include='*.py' --include='*/' --exclude='*' -e "ssh -p 9999" /Users/vince/workspace/tf-deeplab-inception-resnet/ vincentchu@localhost:/home/vincentchu/workspace/tf-deeplab-inception-resnet`
  exec(cmd, (err, stdout, stderr) => {
    console.log(stdout)
    cb()
  })
})

gulp.task('watch', (cb) => {
  gulp.watch('**/*.py', { debounceDelay: 10 }, ['sync'])
})

gulp.task('default', ['watch'])
