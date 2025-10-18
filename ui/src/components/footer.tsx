import { siGithub, siDiscord } from 'simple-icons';

export function Footer() {
  return (
    <footer className="w-full h-10 bg-background">
      <div className="w-full h-full flex items-center justify-center gap-4">
        <div className="flex items-center gap-1 text-xs text-muted-foreground">
          <span className="leading-none">©</span>
          <a
            href="https://xiaoyuanzhu.com"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-foreground transition-colors leading-none"
          >
            小圆猪
          </a>
        </div>

        <a
          href="https://github.com/xiaoyuanzhu-com/homelab-ai-in-docker"
          target="_blank"
          rel="noopener noreferrer"
          className="text-muted-foreground hover:text-foreground transition-colors inline-flex"
          title="GitHub"
        >
          <svg
            role="img"
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
            width="14"
            height="14"
            fill="currentColor"
            className="block"
          >
            <path d={siGithub.path} />
          </svg>
        </a>

        <a
          href="https://discord.gg/Zqrr77UZ"
          target="_blank"
          rel="noopener noreferrer"
          className="text-muted-foreground hover:text-foreground transition-colors inline-flex"
          title="Discord"
        >
          <svg
            role="img"
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
            width="14"
            height="14"
            fill="currentColor"
            className="block"
          >
            <path d={siDiscord.path} />
          </svg>
        </a>
      </div>
    </footer>
  );
}
