# Minki's Blog

Jekyll 기반 개인 블로그입니다.

## 로컬 실행

```bash
cd /Users/minki/minkithub.github.io
bundle install
npm install
npm run dev
```

자산만 다시 생성하려면 아래 명령을 사용하면 됩니다.

```bash
cd /Users/minki/minkithub.github.io
npm run build
bundle exec jekyll build
```

한 번에 검증하려면 아래 명령을 사용하면 됩니다.

```bash
cd /Users/minki/minkithub.github.io
npm run check
```

기본 포트 `4000`이 이미 사용 중이면 아래처럼 바꿔서 실행할 수 있습니다.

```bash
cd /Users/minki/minkithub.github.io
JEKYLL_PORT=4001 npm run dev
```

## 의존성 관리

- Ruby 의존성은 `Gemfile.lock`
- Node 의존성은 `package-lock.json`

오랜만에 다시 실행해도 같은 버전으로 재현되도록 두 lockfile을 함께 관리합니다.

## 블로그 작성 서브에이전트

- 서브에이전트 정의 위치: `.codex/agents/blog-post-writer.toml`
- 목표: 이 블로그의 Jekyll 포스트 형식에 맞춰 초안 작성, 수정, 제목/태그/front matter 정리
- 호출 예시: `[@blog_post_writer](subagent://blog_post_writer)`
